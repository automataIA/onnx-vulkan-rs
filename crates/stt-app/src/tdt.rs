//! TDT (Token-and-Duration Transducer) greedy decoding.
//!
//! Replicates the onnx-asr loop (`_AsrWithTransducerDecoding._decoding` +
//! `NemoConformerTdt._decode`): decoder_joint runs for each encoder frame.
//! It produces `vocab_size` token logits + `num_durations` duration logits;
//! the duration (argmax) indicates how many frames to advance.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;

const ENC_DIM: usize = 1024;
const STATE_LAYERS: usize = 2;
const STATE_DIM: usize = 640;
const MAX_TOKENS_PER_STEP: usize = 10;

pub struct TdtDecoder<'a> {
    session: &'a mut Session,
    vocab_size: usize,
    blank_idx: usize,
}

impl<'a> TdtDecoder<'a> {
    pub fn new(session: &'a mut Session, vocab_size: usize, blank_idx: usize) -> Self {
        Self {
            session,
            vocab_size,
            blank_idx,
        }
    }

    /// `encodings`: encoder output for batch element, layout [ENC_DIM, t_len]
    /// (row-major, as produced by ORT with shape [1, 1024, T]).
    pub fn greedy_decode(&mut self, encodings: &[f32], t_len: usize) -> Result<Vec<usize>> {
        let mut state1 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut state2 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut tokens: Vec<usize> = Vec::new();

        let mut t = 0usize;
        let mut emitted = 0usize;
        while t < t_len {
            // frame t: column t of matrix [ENC_DIM, t_len]
            let mut frame = vec![0f32; ENC_DIM];
            for (i, dst) in frame.iter_mut().enumerate() {
                *dst = encodings[i * t_len + t];
            }

            let last = *tokens.last().unwrap_or(&self.blank_idx) as i32;
            let outputs = self.session.run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(([1usize, ENC_DIM, 1], frame))?,
                "targets" => Tensor::from_array(([1usize, 1], vec![last]))?,
                "target_length" => Tensor::from_array(([1usize], vec![1i32]))?,
                "input_states_1" => Tensor::from_array(([STATE_LAYERS, 1, STATE_DIM], state1.clone()))?,
                "input_states_2" => Tensor::from_array(([STATE_LAYERS, 1, STATE_DIM], state2.clone()))?,
            ])?;

            let (_, logits) = outputs["outputs"].try_extract_tensor::<f32>()?;
            let token_logits = &logits[..self.vocab_size];
            let duration_logits = &logits[self.vocab_size..];

            let token = argmax(token_logits).context("logits vuoti")?;
            let step = argmax(duration_logits).context("logit durata vuoti")?;

            if token != self.blank_idx {
                let (_, s1) = outputs["output_states_1"].try_extract_tensor::<f32>()?;
                let (_, s2) = outputs["output_states_2"].try_extract_tensor::<f32>()?;
                state1.copy_from_slice(s1);
                state2.copy_from_slice(s2);
                tokens.push(token);
                emitted += 1;
            }

            if step > 0 {
                t += step;
                emitted = 0;
            } else if token == self.blank_idx || emitted == MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }
        Ok(tokens)
    }
}

fn argmax(xs: &[f32]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

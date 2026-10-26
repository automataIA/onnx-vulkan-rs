"""Private process boundary: adapters/exporters must obey the caller's timeout."""
import json
from pathlib import Path
import sys

from .prepare import prepare_model, VerificationError
from .pipeline import write_json


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    destination = Path(config.pop("destination"))
    source = Path(config.pop("source"))
    if config.get("adapter"):
        config["adapter"] = Path(config["adapter"])
    quantization = config.pop("quantization", None)
    quality = config.pop("quality", None)
    if quality is not None:
        # Checked before the costly export, optimization and quantization.
        from .quality import check_quantization_compatible, validate_config
        quality = validate_config(quality)
        if quantization is not None:
            check_quantization_compatible(quality, quantization)
    try:
        result = prepare_model(source, destination, **config)
    except VerificationError as error:
        write_json(destination / "preparation.json", error.metadata)
        raise
    if quantization is not None:
        from .quantize import quantize_model
        result["quantization"] = quantize_model(destination / "model.onnx", quantization,
                                               atol=config["atol"], rtol=config["rtol"])
    write_json(destination / "preparation.json", result)
    if quality is not None:
        # Reads preparation.json: a legacy quantization contract is reused, not duplicated.
        from .quality import attach_quality
        result["quality"] = attach_quality(destination, quality)
        write_json(destination / "preparation.json", result)


if __name__ == "__main__":
    main()

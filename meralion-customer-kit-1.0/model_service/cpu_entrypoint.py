"""Run the public benchmark UI and API without loading the model locally."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "model_service.app:production_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
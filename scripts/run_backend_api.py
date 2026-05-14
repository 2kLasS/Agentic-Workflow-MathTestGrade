from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "grade_system.api.app:app",
        host=os.getenv("BACKEND_HOST", "localhost"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()

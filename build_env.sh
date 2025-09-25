# Create virtual env and install dependencies from specified requirements file
REQUIREMENTS_FILE_PATH="$1"

uv venv .venv 
source .venv/bin/activate
uv pip install --no-deps -r ${REQUIREMENTS_FILE_PATH} --extra-index-url https://download.pytorch.org/whl/cu92 --index-strategy unsafe-best-match

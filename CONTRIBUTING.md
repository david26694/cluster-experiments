## Contributing

uv is needed as package manager. If you haven't installed it, run the installation command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project setup

Clone repo and go to the project directory:

```bash
git clone git@github.com:david26694/ab-lab.git
cd ab-lab
```

Create virtual environment and activate it:

```bash
uv venv -p 3.10
source .venv/bin/activate
```

After creating the virtual environment, install the project dependencies:

```bash
make install-dev
```

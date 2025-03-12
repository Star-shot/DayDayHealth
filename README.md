# DayDayHealth
This repo aims to develop a open platform to monitor your health risks, and help you get professional AI's advice.

## 1. Install requirements
```
cd DayDayHealth
pip install -r requirements.txt
```

## 2. Modify configuration

rename config_example.yaml to config.yaml and fill in api_key, base_url, and model_name.
now only support [moonshot](https://platform.moonshot.cn/docs/guide/start-using-kimi-api)

## 3. Run

```
cd src
python src/app.py
```

# TODO
- [ ] add more ML models
- [ ] support more LLMs, include local models
- [ ] create a medical (reasoning) dataset and SFT a model
- [ ] integrate more amazing plots
- [ ] integrate rag database
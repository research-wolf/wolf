# Preparing to use the API

To use Gemini and Palm-2 as APIs, you must be registered with the Google Cloud SDK and own the relevant permissions. For more information, see the links below.

https://ai.google.dev/docs

# Usage

## AI-evaluation for Gemini
```
bash run-gemini.sh
```
## AI-evaluation for PaLM 2
```
bash run-palm2.sh
```
## AI-evaluation for GPT 4
```
bash run-gpt4.sh
```
## Win-rate
```
python win_rate.py \
-e your-ai-evaluation-result.jsonl
```
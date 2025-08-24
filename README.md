Submission for Open Source AI Hackathon #20 - AI Agents

## EEG AI Agent
This project provides an intelligent EEG analysis system that combines automated signal processing with conversational AI to analyze electroencephalography (EEG) data and answer clinical questions. The system is designed to assist medical professionals and researchers by providing both detailed EEG assessments and access to medical knowledge through natural language queries.



## Instructions
### Download Pretrained EEGPT Model

1. **Download the pretrained weights:**
   - Go to: https://figshare.com/s/e37df4f8a907a866df4b
   - Download the ZIP file (974.67 MB)

2. **Extract the model file:**
   - Navigate to: `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
   - This model was trained on mixed datasets (58-channels, 256Hz, 4s time length EEG) using patch size 64

3. **Place in your repo:**
   - Copy `eegpt_mcae_58chs_4s_large4E.ckpt` to your `checkpoint/` directory
   - Final path should be: `checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`

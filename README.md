# Wide Scope Large Multimodal Model Framework for CXR Interpretation

# Overview

In this study, we investigate the enhancement in Chest X-ray (CXR) interpretation performance by Large Multimodal Models (LMMs) through the incorporation of patient- centered health records and the use of structured reports categorized by organs. Specifically, we propose a methodology to integrate patients’ electronic health records (EHR) into model training from a data composition perspective and categorize free-form radiology reports by the corresponding organ. During training, we introduce a modified masked self-attention mechanism that improves the model’s capacity to discern relationships between specific organs and their associated abnormalities. This approach, referred to as the Wide Scope Large Multimodal Model Framework (WoLF), is supported by extensive empirical evidence and analysis, demonstrating significant advancements in the performance of downstream tasks. Our findings provide a promising direction for future research in radiology by enhancing the interpretative capabilities of LMMs.

# Dependencies

Rocky Linux release 8.6 (Green Obsidian)
CUDA Version: 12.1

# Installation

## For Linux:

### Create conda environments

```
conda env update --name wolf --file environment.yaml
```

# Usage

## Access Requirements

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
   - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
   - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
   - Complete the "CITI Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement (DUA) for each project
   - https://physionet.org/sign-dua/mimic-cxr-jpg/2.0.0/
   - https://physionet.org/sign-dua/mimiciv/2.2/

## Data Refomulation

```
cd WoLF/reformulate
```

and please see **reformulate/README.md**

## Training

```
cd WoLF
```

### Stage 1.

```
bash scripts/train/stage1/run.sh
```

### Stage 2.

```
bash scripts/train/stage2/run.sh
```

## Generating Response

### generating answers for VQA

```
bash scripts/generate-response/generate-answers.sh
```

### generating CXR report

```
bash scripts/generate-response/generate-reports.sh
```

## Evaluation

### AI-evaluation

please see **WoLF/ai-eval/README.md**

### Report generation

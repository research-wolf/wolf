# Usage
First you need to obtain the official report of the MIMIC-CXR dataset.
- https://physionet.org/content/mimic-cxr/2.0.0/#files-panel

Second you need to organize the report, run the command below.

See **Anatomical-Specific Knowledge decoupling** in our paper for more on this.

```
python organize_report.py
```

To obtain seed data, see https://github.com/baeseongsu/ehrxqa.

Once you have the seed data, run the command below.
```
python check_data.py
```
Afterward, you will have the final seed data to use for reformulate.

# Thanks to the authors of the papers cited below.
```
@article{bae2023ehrxqa,
  title={EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images},
  author={Bae, Seongsu and Kyung, Daeun and Ryu, Jaehee and Cho, Eunbyeol and Lee, Gyubok and Kweon, Sunjun and Oh, Jungwoo and Ji, Lei and Chang, Eric I and Kim, Tackeun and others},
  journal={arXiv preprint arXiv:2310.18652},
  year={2023}
}
```
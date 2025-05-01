# Datasets description

Development/testing process used the following datasets:
- [BANDON](https://github.com/fitzpchao/BANDON);
- [GVLM](https://github.com/zxk688/GVLM);
- [Hi-UCD](https://github.com/Daisy-7/Hi-UCD-S);
- [LEVIR-CD+](https://github.com/justchenhao/LEVIR);
- [S2Looking](https://github.com/S2Looking/Dataset);
- [SYSU-CD](https://github.com/liumency/SYSU-CD).

Multiple datasets were used to enable multiscale and multi-distribution images, potentially helpful for more generalization. Final training, however, utilized 4 of them, mainly, BANDON, GVLM, Hi-UCD, and LEVIR-CD+. The validation was performed on the datasets that explicitly provided validation sets. Manual validation set creation was not performed to avoid data leakage. Testing was performed with Hi-UCD dataset, being the most challenging for accurate retrieval.

Although some datasets weren’t explicitly created for robust representation learning, the real-world variations they contain effectively mirror the usage conditions of the proposed method.

For reader's convenience, the [Google storage bucket](https://console.cloud.google.com/storage/browser/aerial-disentangled-representations-2025?project=gen-lang-client-0985540815) with these datasets is publicly available for results replication if necessary.
Downloaded datasets have to be placed under [`original` folder](./original). Then, please run `bash scripts/dataset_restructure/all.sh` from the project directory to restructure the datasets for model training.

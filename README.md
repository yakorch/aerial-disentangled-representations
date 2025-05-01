# Disentangled Representation Learning for Sequential Aerial Visual Geopositioning


To download or familiarize yourself with the data, please refer to the [datasets description](./datasets/README.md).

## Motivation
Image retrieval operates on compact image embeddings. We compared two approaches for robust image representation learning building upon the best practices in self-supervised learning. Training is performed on image pairs of the same location under different environmental and/or seasonal conditions.       
 
## Training
Training was performed on a single A100 GPU with 80GB VRAM. Since NT-Xent loss is used for training, a large batch size is required.

The training with the deterministic projector was performed with:
```bash
python -m disentangled_representations.src.training_procedure --batch_size 256 --val_batch_size 256 --num_workers 40 --lr 1e-2 --max_epochs 100 --hidden_features 512 --total_z_dimensionality 128 --temperature 0.1 --w_ntxent 1.0 --w_kl 0.5 --accelerator cuda --anneal_epochs 100
```

The variational approaches included the `--variational` flag, potentially with modified the `--total_z_dimensionality` flag.

[Training logs are provided on Google Drive](https://drive.google.com/drive/folders/1zcvQVzkPhncdBKpMSjqKICrKTPH0VKbK?usp=sharing) with 2 model checkpoints. One can visualize the training logs with:
```bash
tensorboard --logdir=tb_logs/disent_rep
```


### Environment creation
The [`environment.yml`](./environment.yml) is provided. Make sure the `pytorch` is installed with CUDA support for training.


### AI Assistance
Portions of this codebase were generated with the help of OpenAI’s ChatGPT.
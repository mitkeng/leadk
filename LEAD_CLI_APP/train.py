import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import pickle
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import LEAD_Autoencoder, LatentDiscriminator

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Retrain the LEAD Pipeline on new chemical profiles.")
    parser.add_argument('--train_pos', type=str, required=True, help="Path to positive training CSV")
    parser.add_argument('--train_neg', type=str, required=True, help="Path to negative training CSV")
    parser.add_argument('--save_dir', type=str, default="models", help="Directory to save trained model artifacts")
    parser.add_argument('--epochs', type=str, default="250", help="Number of training epochs")
    parser.add_argument('--model_lr', type=float, default=0.0003, help="Learning rate for Autoencoder")
    parser.add_argument('--decoders', type=int, default=5, help="Number of independent subspace decoders")
    parser.add_argument('--adv_weight', type=float, default=0.03, help="Adversarial loss coefficient weight")
    parser.add_argument('--seed', type=int, default=42, help="Global random seed")

    args = parser.parse_args()
    epochs = int(args.epochs)
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n[1/4] Preprocessing Datasets...")
    df_pos = pd.read_csv(args.train_pos)
    df_neg = pd.read_csv(args.train_neg)

    # Identify common features
    dropped_cols = ['Target', 'SMILES', 'Compound']
    features_raw = df_pos.drop(columns=dropped_cols, errors='ignore')
    potential_features = features_raw.select_dtypes(include=[np.number]).columns.tolist()
    features = [f for f in potential_features if f in df_neg.columns]
    input_dim = len(features)
    print(f"-> Identified {input_dim} active continuous features.")

    le = LabelEncoder()
    le.fit(df_pos['Target'].astype(str).str.strip())
    num_classes = len(le.classes_)

    scaler = StandardScaler()
    X_tr_pos_s = scaler.fit_transform(df_pos[features])
    X_tr_neg_s = scaler.transform(df_neg[features])
    y_tr_pos = le.transform(df_pos['Target'].astype(str).str.strip())

    X_tr_pos_t = torch.FloatTensor(X_tr_pos_s)
    X_tr_neg_t = torch.FloatTensor(X_tr_neg_s)
    y_tr_pos_t = torch.LongTensor(y_tr_pos)

    print("\n[2/4] Setting Up Network Architectures...")
    chunks = np.array_split(features, args.decoders)
    group_dims = [len(c) for c in chunks]
    slices = [slice(sum(group_dims[:i]), sum(group_dims[:i+1])) for i in range(args.decoders)]

    model = LEAD_Autoencoder(input_dim, group_dims, num_classes)
    disc_pos = LatentDiscriminator()
    disc_neg = LatentDiscriminator()

    opt_model = optim.AdamW(model.parameters(), lr=args.model_lr, weight_decay=1e-3)
    opt_disc_pos = optim.AdamW(disc_pos.parameters(), lr=0.001)
    opt_disc_neg = optim.AdamW(disc_neg.parameters(), lr=0.003)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=epochs)
    criterion_bce = nn.BCELoss()

    print(f"\n[3/4] Optimizing Pipeline across {epochs} Epochs...")
    for epoch in range(epochs):
        model.train(); disc_pos.train(); disc_neg.train()
        opt_model.zero_grad()
        
        bn_pos, recons_pos, logits_pos = model(X_tr_pos_t)
        ae_loss = sum(nn.MSELoss()(recons_pos[i], X_tr_pos_t[:, slices[i]]) for i in range(args.decoders))
        c_loss = nn.CrossEntropyLoss()(logits_pos, y_tr_pos_t)

        # Latent domain stabilization
        adv_pos = criterion_bce(disc_pos(bn_pos), torch.ones_like(disc_pos(bn_pos)))
        with torch.no_grad():
            bn_neg = model.encoder(X_tr_neg_t)
        adv_neg = criterion_bce(disc_neg(bn_neg), torch.zeros_like(disc_neg(bn_neg)))

        total_loss = c_loss + ae_loss + args.adv_weight * (adv_pos + adv_neg)
        
        if torch.isnan(total_loss):
            print("-> Error: Loss hit NaN. Training aborted.")
            return
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt_model.step()
        scheduler.step()

        # Discriminator 1 Update
        bn_pos_detached = bn_pos.detach()
        opt_disc_pos.zero_grad()
        loss_d_pos = criterion_bce(disc_pos(bn_pos_detached), torch.ones_like(disc_pos(bn_pos_detached))) + \
                     criterion_bce(disc_pos(bn_neg), torch.zeros_like(disc_pos(bn_neg)))
        loss_d_pos.backward()
        opt_disc_pos.step()

        # Discriminator 2 Update
        opt_disc_neg.zero_grad()
        loss_d_neg = criterion_bce(disc_neg(bn_neg), torch.ones_like(disc_neg(bn_neg))) + \
                     criterion_bce(disc_neg(bn_pos_detached), torch.zeros_like(disc_neg(bn_pos_detached)))
        loss_d_neg.backward()
        opt_disc_neg.step()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch+1:03d}/{epochs:03d} | AE+Classifier Loss: {c_loss.item() + ae_loss.item():.4f}")

    print("\n[4/4] Serializing Final Pipeline Models...")
    # Save weights
    torch.save({
        'model_state': model.state_dict(),
        'disc_pos_state': disc_pos.state_dict(),
        'disc_neg_state': disc_neg.state_dict(),
        'config': {
            'input_dim': input_dim,
            'group_dims': group_dims,
            'num_classes': num_classes,
            'features': features
        }
    }, os.path.join(args.save_dir, 'lead_checkpoint.pth'))

    # Save preprocessing transforms
    with open(os.path.join(args.save_dir, 'transforms.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'le': le}, f)

    print(f"-> Successfully completed! Files saved in: {args.save_dir}/")

if __name__ == '__main__':
    main()

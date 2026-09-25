import os
import torch
import pandas as pd
import numpy as np
import argparse
import pickle
from model import LEAD_Autoencoder, LatentDiscriminator, DualDiscriminatorShieldAgent

def main():
    parser = argparse.ArgumentParser(description="Run inference on raw target features.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV containing candidate features")
    parser.add_argument('--checkpoint', type=str, default="models/lead_checkpoint.pth", help="Path to saved model checkpoint")
    parser.add_argument('--transforms', type=str, default="models/transforms.pkl", help="Path to saved scaler & label encoder transform pickle")
    parser.add_argument('--output', type=str, default="predictions_output.csv", help="Path to save calibrated log output file")
    parser.add_argument('--penalty_max', type=float, default=6.0, help="Dual-discriminator maximum temperature penalty limit")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint) or not os.path.exists(args.transforms):
        print(f"-> Error: Checkpoint files not found. Run train.py first to establish directory targets.")
        return

    # Load serialized transformers
    with open(args.transforms, 'rb') as f:
        transforms = pickle.load(f)
    scaler = transforms['scaler']
    le = transforms['le']

    # Load checkpoint variables
    ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    cfg = ckpt['config']
    features = cfg['features']

    # Initialize models
    model = LEAD_Autoencoder(cfg['input_dim'], cfg['group_dims'], cfg['num_classes'])
    disc_pos = LatentDiscriminator()
    disc_neg = LatentDiscriminator()

    model.load_state_dict(ckpt['model_state'])
    disc_pos.load_state_dict(ckpt['disc_pos_state'])
    disc_neg.load_state_dict(ckpt['disc_neg_state'])

    # Read input records
    df_in = pd.read_csv(args.input)
    missing_feats = [f for f in features if f not in df_in.columns]
    if missing_feats:
        print(f"-> Error: Input is missing necessary feature columns: {missing_feats}")
        return

    X_scaled = scaler.transform(df_in[features])
    X_t = torch.FloatTensor(X_scaled)

    # Predict via calibrated Shield Agent
    shield = DualDiscriminatorShieldAgent(model, disc_pos, disc_neg, penalty_max=args.penalty_max)
    probs = shield.predict_shielded(X_t)

    # Format results output
    logs = []
    for idx, row in df_in.iterrows():
        sample_probs = probs[idx]
        top_5_idx = np.argsort(sample_probs)[::-1][:5]
        top_5_labels = le.inverse_transform(top_5_idx)
        top_5_scores = sample_probs[top_5_idx]

        predictions_str = " | ".join([f"{lbl} ({scr:.4f})" for lbl, scr in zip(top_5_labels, top_5_scores)])
        
        record = {}
        if 'Compound' in df_in.columns:
            record['Compound'] = row['Compound']
        if 'Target' in df_in.columns:
            record['True Target'] = row['Target']
            record['True Target Captured'] = row['Target'].strip() in top_5_labels

        record['Top 5 Predictions (Confidence)'] = predictions_str
        logs.append(record)

    df_out = pd.DataFrame(logs)
    df_out.to_csv(args.output, index=False)
    print(f"\n✅ Inference Complete! Results successfully saved to: '{args.output}' ({len(df_out)} rows)")

if __name__ == '__main__':
    main()

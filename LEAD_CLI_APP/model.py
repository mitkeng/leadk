import torch
import torch.nn as nn
import numpy as np

class LEAD_Autoencoder(nn.Module):
    """
    Dynamic Multi-Decoder Autoencoder
    Maps biological continuous features to a 32D latent bottleneck
    reconstructed by multiple independent decoders.
    """
    def __init__(self, input_dim, group_dims, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoders = nn.ModuleDict({
            str(i): nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, dim))
            for i, dim in enumerate(group_dims)
        })
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        recons = [self.decoders[str(i)](z) for i in range(len(self.decoders))]
        return z, recons, self.classifier(z)

class LatentDiscriminator(nn.Module):
    """
    Latent space discriminator to verify structural authenticity
    of candidate drug embeddings.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class DualDiscriminatorShieldAgent:
    """
    Inference Shield calibration agent.
    Applies temperature scaling to classification logits based on the
    relative confidence scores of positive and negative latent discriminators.
    """
    def __init__(self, trained_model, d_pos, d_neg, penalty_max=6.0):
        self.model = trained_model
        self.d_pos = d_pos
        self.d_neg = d_neg
        self.penalty_max = penalty_max

    def predict_shielded(self, X_tensor):
        self.model.eval()
        self.d_pos.eval()
        self.d_neg.eval()

        with torch.no_grad():
            z, _, raw_logits = self.model(X_tensor)
            logits = raw_logits.cpu().numpy()
            pos_scores = self.d_pos(z).cpu().numpy().flatten()
            neg_scores = self.d_neg(z).cpu().numpy().flatten()

        calibrated_probabilities = []
        for idx in range(X_tensor.size(0)):
            penalty = 1.0 + (neg_scores[idx] / (pos_scores[idx] + 1e-7))
            penalty = np.clip(penalty, 1.0, self.penalty_max)

            shifted_logits = logits[idx] - np.max(logits[idx])
            exp_logits = np.exp(shifted_logits / penalty)
            probs = exp_logits / (np.sum(exp_logits) + 1e-9)
            calibrated_probabilities.append(probs)

        return np.array(calibrated_probabilities)

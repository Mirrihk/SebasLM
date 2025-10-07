using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SebasLM.Core.Model.Normalizations;
using SebasLM.Core.Model.Blocks;
using TorchSharp.Modules;

namespace SebasLM.Core.Model
{

    /// <summary>
    /// TinyGPT: A small GPT-style language model for demonstration and testing.
    /// /// Features:
    /// - Embedding layer for token and positional embeddings.
    /// - Stack of Transformer blocks with self-attention and SwiGLU feed-forward.
    /// - RMSNorm for normalization.
    /// - Final linear layer to project to vocabulary size.
    /// - Causal masking in attention to prevent future token access.
    /// - Dropout for regularization.
    /// - Configurable model size via parameters.
    /// tokens -> tokemEmbed + posEmbed -> [TransformerBlock x N] -> RMSNorm -> lmHead -> logits
    ///  
    /// </summary>

    public sealed class TinyGPT : Module
    {
        private readonly long vocabSize, dModel, nLayers, nHeads, maxT;
        private readonly Module tokenEmbed, posEmbed, drop;
        private readonly ModuleList blocks;
        private readonly RMSNorm finalNorm;
        private readonly Module lmHead;

        public TinyGPT(string name, long vocabSize, long dModel = 128, long nLayers = 2, long nHeads = 4, long maxT = 512, double pDrop = 0.1)
            : base(name)
        {
            this.vocabSize = vocabSize;
            this.dModel = dModel;
            this.nLayers = nLayers;
            this.nHeads = nHeads;
            this.maxT = maxT;

            tokenEmbed = Embedding(vocabSize, dModel);
            posEmbed = Embedding(maxT, dModel);
            drop = Dropout(pDrop);

            blocks = new ModuleList();
            for (int i = 0; i < nLayers; i++)
            {
                var block = new TransformerBlock($"{name}.block{i}", dModel, nHeads, ffnMult: 4, pAttn: pDrop, pProj: pDrop, pFF: pDrop);
                blocks.Append(block);
            }

            finalNorm = new RMSNorm($"{name}.rms_final", dModel);
            lmHead = Linear(dModel, vocabSize);

            RegisterComponents();
        }

        public override TorchTensor forward(TorchTensor tokens)
        {
            // tokens: [B,T] long
            var (B, T) = (tokens.shape[0], tokens.shape[1]);
            if (T > maxT)
                throw new ArgumentException($"Input sequence length T={T} exceeds model maxT={maxT}.");

            // Token and positional embeddings
            var tokEmb = tokenEmbed.forward(tokens); // [B,T,dModel]
            var posIds = torch.arange(0, T, dtype: torch.int64, device: tokens.device).unsqueeze(0).expand(B, T);
            var posEmb = posEmbed.forward(posIds);   // [B,T,dModel]

            var x = tokEmb + posEmb;
            x = drop.forward(x);

            // Transformer blocks
            foreach (var block in blocks)
            {
                x = block.forward(x);
            }

            // Final norm and LM head
            x = finalNorm.forward(x);  // [B,T,dModel]
            var logits = lmHead.forward(x); // [B,T,vocabSize]

            return logits;
        }
    }
}
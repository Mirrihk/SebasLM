using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SebasLM.Core.Model.Blocks;
using SebasLM.Core.Model.Normalizations;


namespace SebasLM.Core.Model
{
    public sealed class TinyGPT : Module<Tensor, Tensor>
    {
        private readonly long vocabSize;
        private readonly long modelDim;
        private readonly int  numLayers;
        private readonly int  numHeads;
        private readonly int  maxSeqLen;
        private readonly double dropout;

        // Modules
        private readonly Module<Tensor, Tensor> tokEmb;     // nn.Embedding
        private readonly Module<Tensor, Tensor> posEmb;     // nn.Parameter 
        private readonly ModuleList<TransformerBlock> blocks;
        private readonly Module<Tensor, Tensor> lnFinal;    // LayerNorm
        private readonly Module<Tensor, Tensor> lmHead;     // Linear to vocab

        public TinyGPT(
            long vocabSize,
            long modelDim,
            int  numLayers,
            int  numHeads,
            int  maxSeqLen,
            double dropout = 0.0,
            double ffnMult = 4.0,               // convenience: compute hidden from mult
            string name = "tiny_gpt")
            : base(name)
        {
            if (modelDim % numHeads != 0)
                throw new ArgumentException("modelDim must be divisible by numHeads.");

            this.vocabSize = vocabSize;
            this.modelDim  = modelDim;
            this.numLayers = numLayers;
            this.numHeads  = numHeads;
            this.maxSeqLen = maxSeqLen;
            this.dropout   = dropout;

            // 1) Token embedding: indices -> modelDim
            tokEmb = Embedding(vocabSize, modelDim);

            // 2) Positional embedding parameter [maxSeqLen, modelDim]
            posEmb = Embedding(maxSeqLen, modelDim);
            // 3) Blocks
            blocks = new ModuleList<TransformerBlock>();
            var ffnHidden = (long)Math.Round(ffnMult * modelDim);

            for (int i = 0; i < numLayers; i++)
            {
                blocks.Add(new TransformerBlock(modelDim, numHeads, ffnHidden, dropout: dropout, name: $"block{i}"));
            }

            // 4) Final norm + LM head
            lnFinal = LayerNorm(modelDim);
            lmHead  = Linear(modelDim, vocabSize, hasBias: false);

            RegisterComponents();
        }

        public override Tensor forward(Tensor tokenIds)
        {
            using var scope = torch.NewDisposeScope();
            // tokenIds: [B, T] (dtype: Int64)

            var B = tokenIds.shape[0];
            var T = tokenIds.shape[1];

            // [B, T, C]
            var x = tokEmb.forward(tokenIds);

            // Add positions: broadcast posEmb[0:T]
            var positions = torch.arange(T, dtype: ScalarType.Int64, device: tokenIds.device);
            var pos = posEmb.forward(positions).unsqueeze(0); // [1, T, C]
            x = x + pos;
            // Transformer blocks
            for (int i = 0; i < numLayers; i++)
            {
                x = blocks[i].forward(x);
            }

            x = lnFinal.forward(x);

            // Project to vocab
            var logits = lmHead.forward(x);             // [B, T, vocab]
            return logits.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                tokEmb?.Dispose();
                posEmb?.Dispose();
                lnFinal?.Dispose();
                lmHead?.Dispose();
                if (blocks is not null)
                {
                    foreach (var b in blocks) b?.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }
}

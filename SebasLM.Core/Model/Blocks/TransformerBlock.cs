using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Core.Model.Blocks
{
    public sealed class TransformerBlock : Module<Tensor, Tensor>
    {
        private readonly int heads;
        private readonly long modelDim;
        private readonly bool preNorm;

        private readonly Module<Tensor, Tensor> ln1;
        private readonly Module<Tensor, Tensor> ln2;
        private readonly Module<Tensor, Tensor> drop;

        private readonly SelfAttention attn;
        private readonly FeedForwardSwiGLU ffn;

        public TransformerBlock(long modelDim, int numHeads, long ffnHidden, double dropout = 0.0, bool preNorm = true, string name = "transformer_block")
            : base(name)
        {
            if (modelDim % numHeads != 0)
                throw new ArgumentException("modelDim must be divisible by numHeads.");

            this.modelDim = modelDim;
            this.heads    = numHeads;
            this.preNorm  = preNorm;

            ln1  = LayerNorm(modelDim);
            ln2  = LayerNorm(modelDim);
            drop = Dropout(dropout);

            // IMPORTANT: constructor arity matches our earlier classes (3 args each)
            attn = new SelfAttention(modelDim, numHeads, $"{name}.attn");
            ffn  = new FeedForwardSwiGLU(modelDim, ffnHidden, $"{name}.ffn");

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            using var scope = torch.NewDisposeScope();

            if (preNorm)
            {
                // Pre-norm: x = x + Attn(LN(x))
                var a = attn.forward(ln1.forward(x));
                x = x + drop.forward(a);

                // x = x + FFN(LN(x))
                var f = ffn.forward(ln2.forward(x));
                x = x + drop.forward(f);
            }
            else
            {
                // Post-norm variant (less common these days)
                var a = attn.forward(x);
                x = ln1.forward(x + drop.forward(a));

                var f = ffn.forward(x);
                x = ln2.forward(x + drop.forward(f));
            }

            return x.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                ln1?.Dispose(); ln2?.Dispose(); drop?.Dispose();
                attn?.Dispose(); ffn?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}

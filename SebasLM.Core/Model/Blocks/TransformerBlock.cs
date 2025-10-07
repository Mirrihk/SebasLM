using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SebasLM.Core.Model.Normalizations;

namespace SebasLM.Core.Model.Blocks
{
    /// <summary>
    /// Pre-norm transformer block:
    /// x = x + SelfAttention( RMSNorm(x) )
    /// x = x + FeedForward(  RMSNorm(x) )
    /// </summary>
    public sealed class TransformerBlock : Module
    {
        private readonly RMSNorm norm1;
        private readonly RMSNorm norm2;
        private readonly SelfAttention attn;
        private readonly FeedForwardSwiGLU ffn;

        public TransformerBlock(string name, long dModel, long nHeads, long ffnMult = 4, double pAttn = 0.0, double pProj = 0.0, double pFF = 0.0)
            : base(name)
        {
            norm1 = new RMSNorm($"{name}.rms1", dModel);
            norm2 = new RMSNorm($"{name}.rms2", dModel);
            attn = new SelfAttention($"{name}.attn", dModel, nHeads, pAttn, pProj);
            ffn  = new FeedForwardSwiGLU($"{name}.ffn", dModel, ffnMult, pFF);
            RegisterComponents();
        }

        public override TorchTensor forward(TorchTensor x)
        {
            x = x + attn.forward(norm1.forward(x));
            x = x + ffn.forward(norm2.forward(x));
            return x;
        }
    }
}

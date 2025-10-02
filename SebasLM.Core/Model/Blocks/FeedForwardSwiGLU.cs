using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using SebasLM.Core.Model.Activations;
using System.Globalization;

namespace SebasLM.Core.Model.Blocks
{
    /*
    * 
    * FFN block using SwiGLU Activation.
    * y = Linear2(SwiGLU(Linear1(x))),with dropout after SwiGLU. 
    * Shapes: [B,T,d] -> [B, T, d] 
    *
    */

    public sealed class FeedForwardSwiGLU : Module
    {
        private readonly Module projectIn;
        private readonly Module projectOut;
        private readonly Module dropout;
        private readonly long d;
        private readonly long hidden;

        public FeedForwardSwiGLU(string name, long dModel, long hiddenMult, double dropoutProb = 0.0) : base(name)
        {
            d = dModel;
            hidden = hiddenMult * dModel;
            projectIn = Linear(dModel, 2 * hidden);
            projectOut = Linear(hidden, dModel);
            dropout = Dropout(dropoutProb);
            RegisterComponents();
        }
        public override TorchTensor forward(TorchTensor x)
        {
            // x: [B, T, d]
            var h = projectIn.forward(x); // split last dim into 2 halves
            var (a, b) = h.split((long)hidden, dim: -1);

            // Apply SwiGLU: SiLU(a) * b
            var y = functional.silu(a) * b;

            y = projectIn.forward(y);
            y = dropout.forward(y);
            return y;
        }
    }
}
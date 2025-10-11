using System;
using TorchSharp;
using  TorchSharp.Modules;
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

    // 1) Inherit the GENERIC Module so `forward` exists
    public sealed class FeedForwardSwiGLU : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> projIn;   // Linear -> produces 2x hidden (for gate + value)
        private readonly Module<Tensor, Tensor> projOut;  // Linear -> back to model dim

        public FeedForwardSwiGLU(long modelDim, long hiddenDim, string name = "ffn_swiglu") : base(name)
        {
            // typical SwiGLU FFN: in -> Linear(2*hidden) -> chunk -> silu(gate)*val -> Linear(out)
            projIn  = Linear(modelDim, 2 * hiddenDim, hasBias: true);
            projOut = Linear(hiddenDim, modelDim, hasBias: true);
            RegisterComponents(); // 2) always register your submodules
        }

        // 3) Override forward(Tensor)
        public override Tensor forward(Tensor x)
        {
            using var scope = NewDisposeScope();

            var y = projIn.forward(x);                     // [B, T, 2H] or [N, 2H] depending on shape
            var chunks = y.chunk(2, dim: -1);              // split into gate/value along last dim
            var gate = functional.silu(chunks[0]);         // SwiGLU = SiLU(gate) * value
            var val  = chunks[1];
            var act  = gate.mul_(val);

            var outy = projOut.forward(act);
            return outy.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) { projIn?.Dispose(); projOut?.Dispose(); }
            base.Dispose(disposing);
        }
    }
}
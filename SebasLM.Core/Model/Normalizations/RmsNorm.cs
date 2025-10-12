using System;
using TorchSharp;
using TorchSharp.Modules;           // Parameter
using static TorchSharp.torch;      // Tensor, torch.*
using static TorchSharp.torch.nn;   // (not strictly needed here, but handy)

namespace SebasLM.Core.Model.Normalizations
{
    // Root Mean Square Layer Norm (no mean-centering), as used in many LLMs.
    // Normalizes across the last dimension.
    public sealed class RMSNorm : Module<Tensor, Tensor>
    {
        private readonly long featureDim;
        private readonly double eps;
        private readonly Parameter weight; // learnable scale (gamma)

        // normalizedShape == size of last dimension (e.g., modelDim)
        public RMSNorm(long normalizedShape, double eps = 1e-5, string name = "rmsnorm")
            : base(name)
        {
            featureDim = normalizedShape;
            this.eps = eps;

            // gamma initialized to ones
            weight = Parameter(torch.ones(new long[] { featureDim }));

            RegisterComponents(); // important for parameters/buffers registration
        }

        public override Tensor forward(Tensor x)
        {
            using var scope = torch.NewDisposeScope();

            // x: [*, featureDim]
            // rms = sqrt(mean(x^2, dim=-1, keepdim=true) + eps)
            var ms = torch.mean(x.pow(2), new long[] { -1 }, true);  
            var rms  = (ms + eps).sqrt_();                   // in-place sqrt is fine here
            var y    = x / rms;                              // broadcast over last dim

            // scale by learnable gamma (weight)  -> broadcast over last dim
            var outy = y * weight;

            return outy.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) { weight?.Dispose(); }
            base.Dispose(disposing);
        }
    }
}

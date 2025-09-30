using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Core.Model.Normalizations
{
    /// <Summary>
    /// RMSNorm (Root Mean Square Layer Normalization).
    /// 
    /// Instead of subtracting the mean (as in LayerNorm),
    /// this normalizes only by the Root Mean Square (RMS) of the activations:
    /// 
    ///   y = (x / RMS(x)) * weight
    ///   RMS(x) = sqrt(mean(x^2) + eps)
    /// 
    /// where 'weight' is a learnable scale vector (per hidden dim).
    /// 
    /// Benefits:
    ///   - Lighter than LayerNorm (no mean subtraction).
    ///   - Stable for very deep LLMs (used in LLaMA, Falcon, etc.).
    ///   - Keeps scale-invariance without affecting mean.
    /// 
    /// Shape:
    ///   Input:  [*, d]  (any leading dims, last dim = hidden size)
    ///   Output: [*, d]
    /// </summary>
    public sealed class RMSNorm : Module
    {
        private readonly Parameter weight;
        private readonly double eps;

        /// <summary>
        /// Create an RMSNorm layer.
        /// </summary>
        /// <param name="name">Module name (for TorchSharp tracking).</param>
        /// <param name="dim">Hidden size (last dimension of input).</param>
        /// <param name="eps">Small epsilon for numerical stability (default 1e-6).</param>
        public RMSNorm(string name, long dim, double eps = 1e-6) : base(name)
        {
            this.eps = eps;

            // Learnable scale parameter, initialized to ones.
            weight = torch.nn.Parameter(torch.ones(dim, dtype: float32));
            RegisterComponents();
        }

        /// <summary>
        /// Forward pass of RMSNorm.
        /// </summary>
        /// <param name="x">Input tensor of shape [*, d].</param>
        /// <returns>Normalized tensor of shape [*, d].</returns>
        public override Tensor forward(Tensor x)
        {
            // Compute RMS across last dimension:
            // RMS(x) = sqrt(mean(x^2, dim=-1) + eps), keep same shape
            var rms = x.pow(2).mean(dim: -1, keepdim: true).add(eps).sqrt();

            // Normalize by RMS
            var normed = x / rms;

            // Apply learnable scale (broadcasted across last dim)
            return normed * weight;
        }
    }
}

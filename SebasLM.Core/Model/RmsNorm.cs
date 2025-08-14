using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Model
{
    public sealed class RMSNorm : Module
    {
        private readonly Tensor weight;
        private readonly double eps;
        public RMSNorm(string name, long dim, double eps = 1e-6) : base(name)
        {
            this.eps = eps;
            weight = torch.ones(dim, dtype: float32);
            RegisterComponents();
        }
        public override Tensor forward(Tensor x)
        {
            // x: [*, d]
            var rms = x.pow(2).mean(dim: -1, keepdim: true).add(eps).sqrt();
            var outT = x / rms;
            return outT * weight;
        }
    }
}

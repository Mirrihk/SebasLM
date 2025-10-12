using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Core.Model.Blocks
{
    public sealed class SelfAttention : Module<Tensor, Tensor>
    {
        private readonly int heads;
        private readonly long headDim;
        private readonly Module<Tensor, Tensor> qProj;
        private readonly Module<Tensor, Tensor> kProj;
        private readonly Module<Tensor, Tensor> vProj;
        private readonly Module<Tensor, Tensor> oProj;

        public SelfAttention(long modelDim, int numHeads, string name = "self_attn")
            : base(name)
        {
            if (modelDim % numHeads != 0)
                throw new ArgumentException("modelDim must be divisible by numHeads.");

            heads   = numHeads;
            headDim = modelDim / numHeads;

            // Make sure these are Module<Tensor,Tensor>
            qProj = Linear(modelDim, modelDim, hasBias: true);
            kProj = Linear(modelDim, modelDim, hasBias: true);
            vProj = Linear(modelDim, modelDim, hasBias: true);
            oProj = Linear(modelDim, modelDim, hasBias: true);

            RegisterComponents();
        }

        // Override EXACTLY one-arg forward to satisfy the base class.
        public override Tensor forward(Tensor x) => Forward(x, null);

        // Optional mask-friendly entry point (not an override).
        // attnMask expected broadcastable to [B, heads, T_q, T_k] (e.g., [B,1,1,T] or [1,1,T,T])
        public Tensor Forward(Tensor x, Tensor? attnMask)
        {
            using var scope = torch.NewDisposeScope();

            // Shapes assumed: x: [B, T, C]
            var B = x.shape[0];
            var T = x.shape[1];
            var C = x.shape[2];

            var q = qProj.forward(x); // [B, T, C]
            var k = kProj.forward(x);
            var v = vProj.forward(x);

            // Reshape to heads
            q = q.view(B, T, heads, headDim).transpose(1, 2); // [B, H, T, d]
            k = k.view(B, T, heads, headDim).transpose(1, 2); // [B, H, T, d]
            v = v.view(B, T, heads, headDim).transpose(1, 2); // [B, H, T, d]

            // Scaled dot-product attention
            var scale = 1.0 / Math.Sqrt((double)headDim);
            var scores = torch.matmul(q, k.transpose(-2, -1)) * scale; // [B, H, T, T]

            if (attnMask is not null)
            {
                // Expect mask with 1=keep, 0=block (broadcastable)
                // Add a large negative number to masked positions
                var negInf = (-1e9).ToScalar();
                scores = scores + (attnMask.eq(0).to_type(scores.dtype)).mul(negInf);
            }

            var probs = functional.softmax(scores, dim: -1);          // [B, H, T, T]
            var ctx   = torch.matmul(probs, v);                       // [B, H, T, d]

            // Merge heads: [B, T, H, d] -> [B, T, C]
            ctx = ctx.transpose(1, 2).contiguous().view(B, T, C);

            var outy = oProj.forward(ctx); // [B, T, C]
            return outy.MoveToOuterDisposeScope();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) { qProj?.Dispose(); kProj?.Dispose(); vProj?.Dispose(); oProj?.Dispose(); }
            base.Dispose(disposing);
        }
    }
}

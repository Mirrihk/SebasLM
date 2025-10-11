using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Core.Model.Blocks
{
    /// <summary>
    /// Multi-head causal self-attention.
    /// Pre-norm block will wrap this; here we only do attention math.
    /// Shapes: x [B,T,d] -> [B,T,d]
    /// </summary>
    public sealed class SelfAttention : Module
    {
        private readonly long dModel, nHeads, headDim;
        private readonly Module qProj, kProj, vProj, oProj;
        private readonly Module dropoutAttn, dropoutProj;
        private readonly double scale;

        public SelfAttention(string name, long dModel, long nHeads, double pAttn = 0.0, double pProj = 0.0)
            : base(name)
        {
            this.dModel = dModel;
            this.nHeads = nHeads;
            headDim = dModel / nHeads;
            scale = 1.0 / Math.Sqrt(headDim);

            qProj = Linear(dModel, dModel);
            kProj = Linear(dModel, dModel);
            vProj = Linear(dModel, dModel);
            oProj = Linear(dModel, dModel);

            dropoutAttn = Dropout(pAttn);
            dropoutProj = Dropout(pProj);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x, Tensor? mask = null)
{
    var (B, T, D) = (x.shape[0], x.shape[1], x.shape[2]);

    var q = qProj.forward(x);
    var k = kProj.forward(x);
    var v = vProj.forward(x);

    q = q.view(B, T, nHeads, headDim).transpose(1, 2);
    k = k.view(B, T, nHeads, headDim).transpose(1, 2);
    v = v.view(B, T, nHeads, headDim).transpose(1, 2);

    var scores = q.matmul(k.transpose(-2, -1)) * scale;

    if (mask is null)
    {
        var causal = torch.tril(torch.ones(new long[] { T, T }, dtype: torch.int8, device: x.device));
        var negInf = torch.tensor(double.NegativeInfinity, dtype: scores.dtype, device: x.device);
        var where = causal.eq(0).unsqueeze(0).unsqueeze(0);
        scores = scores.masked_fill(where, negInf);
    }
    else
    {
        scores = scores + mask;
    }

    var attn = functional.softmax(scores, dim: -1);
    attn = dropoutAttn.forward(attn);

    var ctx = attn.matmul(v);
    ctx = ctx.transpose(1, 2).contiguous().view(B, T, D);

    var outProj = oProj.forward(ctx);
    outProj = dropoutProj.forward(outProj);
    return outProj;
}
       

    }
}

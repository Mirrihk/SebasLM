using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SebasLM.Core.Model.Blocks
{
    /*
    *
    * Self-Attention block.
    * Pre-norm, with residual connection.
    * Shapes: [B,T,d] -> [B, T, d]q
    * B: batch size
    *
    */
    public sealed class SelfAttention : Module
    {
        private readonly long dModel, nHeads, headDim;
        private readonly Module qProj, kProj, vProj, outProj;
        private readonly Module dropoutAttn, dropoutProj;
        private readonly double scale;

        public SelfAttention(string name, long dModel, long nHeads, double pAttn = 0.0, double pProj = 0.0) : base(name)
        {
            this.dModel = dModel;
            this.nHeads = nHeads;
            headDim = dModel / nHeads;
            scale = 1.0 / Math.Sqrt(headDim);

            qProj = Linear(dModel, dModel);
            kProj = Linear(dModel, dModel);
            vProj = Linear(dModel, dModel);
            outProj = Linear(dModel, dModel);

            dropoutAttn = Dropout(pAttn);
            dropoutProj = Dropout(pProj);

            RegisterComponents();
        }

        public override TorchTensor forward(TorchTensor x, TorchTensor? mask = null)
        {
            var (B, T, D) = (x.shape[0], x.shape[1], x.shape[2]);

            var q = qProj.forward(x);
            var k = qProj.forward(x);
            var v = qProj.forward(x);

            // reshape to (B, nHeads, T, headDim)
            q = q.view(B, T, nHeads, headDim).transpose(1, 2);
            k = k.view(B, T, nHeads, headDim).transpose(1, 2);
            v = v.view(B, T, nHeads, headDim).transpose(1, 2);

            // score = (B, nHeads, T, T)
            var scores = q.matmul(k.transpose(-2, -1)) * scale;

            if (mask is null)
            {
                var casual = torch.tril(torch.ones(new long[] { T, T }, dtype: torch.int8, device: x.device));
                var negInf = torch.tensor(double.NegativeInfinity, dtype: scores.dtype, device: x.device);
                var where = casual.eq(0).unsqueeze(0).unsqueeze(0);
                scores = scores.where(where, negInf);
            }
            else
            {
                scores = scores + mask;
            }

            var attn = functional.softmax(scores, -1);
            attn = dropoutAttn.forward(attn);

            var ctx = attn.matmul(v); // (B, nHeads, T, headDim)

            // merge heads : [B, T, headDim] -> [B, T, dModel]
            ctx = ctx.transpose(1, 2).contiguous().view(B, T, D);

            var outProj = oProj.forward(ctx);
            outProj = dropoutProj.forward(outProj);
            return outProj;
        }


    }






}







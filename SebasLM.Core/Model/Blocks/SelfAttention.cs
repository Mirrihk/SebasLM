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


    }






}







// SebasLM.Train/Program.cs
// Build/Run: dotnet run -c Release --project SebasLM.Train

using System;
using System.Linq;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SebasLM.Core.Model; // TinyGPT

namespace SebasLM.Train
{
    // ------------------------------------------------------------
    // Device selection
    // ------------------------------------------------------------
    static class DeviceUtil
    {
        public static Device Best => cuda.is_available() ? CUDA : CPU;
        private static Device CUDA => torch.CUDA;
        private static Device CPU  => torch.CPU;
    }

    // ------------------------------------------------------------
    // Simple printable-ASCII tokenizer (chars 32..126)
    // vocab = 95, maps any out-of-range char to id 0 (space)
    // ------------------------------------------------------------
    sealed class TinyCharTokenizer
    {
        public int VocabSize => 95;

        private readonly char[] _idx2ch =
            Enumerable.Range(32, 95).Select(i => (char)i).ToArray();

        private readonly System.Collections.Generic.Dictionary<char, int> _ch2idx;

        public TinyCharTokenizer()
        {
            _ch2idx = _idx2ch.Select((c, i) => (c, i))
                              .ToDictionary(x => x.c, x => x.i);
        }

        public Tensor Encode(string s, Device device)
        {
            var ids = s.Select(c => _ch2idx.TryGetValue(c, out var id) ? id : 0).ToArray();
            return torch.tensor(ids, dtype: ScalarType.Int64, device: device);
        }

        public string Decode(Tensor ids)
        {
            var data = ids.to_type(ScalarType.Int64).cpu().data<long>().ToArray();
            var sb = new StringBuilder();
            foreach (var id in data)
            {
                int safe = (int)Math.Clamp(id, 0, _idx2ch.Length - 1);
                sb.Append(_idx2ch[safe]);
            }
            return sb.ToString();
        }
    }

    // ------------------------------------------------------------
    // Training program (TinyGPT + real text + sampling)
    // ------------------------------------------------------------
    class Program
    {
        static void Main()
        {
            torch.random.manual_seed(0);

            var device = DeviceUtil.Best;
            Console.WriteLine($"Device: {device.type}");

            // --- Tokenizer & corpus ---
            var tok = new TinyCharTokenizer();
            // Personalize this string or load from a file for a better demo:
            var corpus = "hello from sebaslm — a tiny transformer in c# with torchsharp!\n";
            const int blockSize = 128;
            const int batchSize = 16;

            // --- Model ---
            // Use tokenizer's vocab and keep a small config for speed.
            long vocab = tok.VocabSize;
            long d = 256, heads = 8, layers = 4, maxT = blockSize;

            var model = new TinyGPT(
                "sebaslm_tinygpt", // name
                vocab,             // vocabSize
                d,                 // dModel
                heads,             // nHeads
                layers,            // nLayers
                maxT,              // max sequence length
                0.0                // pDrop
            ).to(device);

            // --- Optimizer ---
            var optim = torch.optim.AdamW(
    model.parameters(),
    3e-4,   // lr
    0.9,    // betas1
    0.95,   // betas2
    1e-8,   // eps
    0.01    // weight_decay
);

            // --- Encode corpus; duplicate if too short to form a batch ---
            var data = tok.Encode(corpus, device);
            if (data.shape[0] < blockSize + 1)
                data = torch.cat(new Tensor[] { data, data }, 0);

            // --- Tiny training loop (char-level next-token prediction) ---
            int steps = 200;
            model.train();
            for (int step = 1; step <= steps; step++)
            {
                var (x, y) = SampleBatch(data, batchSize, blockSize, device);
                optim.zero_grad();

                // logits: [B,T,V]
                var logits = model.forward(x);

                // Cross-entropy over flattened time+batch
                var loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, (long)vocab),
                    y.reshape(-1)
                );

                loss.backward();
                optim.step();

                if (step % 25 == 0)
                    Console.WriteLine($"step {step}/{steps} | loss {loss.ToSingle():0.000}");
            }

            // --- Generation demo ---
            model.eval();
            var prompt = "hello";
            var start = tok.Encode(prompt, device).unsqueeze(0); // [1,T]
            var gen = Generate(model, start, maxNewTokens: 120, topK: 30, temperature: 0.9);

            Console.WriteLine("\n=== Generated ===");
            Console.WriteLine(tok.Decode(gen[0]));
        }

        // ------------------------------------------------------------
        // Batch sampler: draws random contiguous chunks from 1D token array
        // Returns:
        //   x: [B, block]  inputs
        //   y: [B, block]  next-token targets (shifted by +1)
        // ------------------------------------------------------------
        static (Tensor x, Tensor y) SampleBatch(Tensor data, int batch, int block, Device device)
        {
            var rnd = new Random();
            var Xs = new System.Collections.Generic.List<Tensor>();
            var Ys = new System.Collections.Generic.List<Tensor>();
            long N = data.shape[0];

            for (int b = 0; b < batch; b++)
            {
                int s = rnd.Next(0, (int)Math.Max(1, N - block - 1));
                int e = s + block;

                var xSlice = data.index(new TensorIndex[] { TensorIndex.Slice(s, e) }).unsqueeze(0);
                var ySlice = data.index(new TensorIndex[] { TensorIndex.Slice(s + 1, e + 1) }).unsqueeze(0);

                Xs.Add(xSlice);
                Ys.Add(ySlice);
            }

            var X = torch.cat(Xs.ToArray(), dim: 0).to(device).to_type(ScalarType.Int64);
            var Y = torch.cat(Ys.ToArray(), dim: 0).to(device).to_type(ScalarType.Int64);
            return (X, Y);
        }

        // ------------------------------------------------------------
        // Autoregressive generation
        //   - Starts from 'idx' [B, T0]
        //   - Appends tokens one-by-one using model logits
        //   - Supports temperature and top-k sampling
        // Returns: [B, T0 + maxNewTokens]
        // ------------------------------------------------------------
        static Tensor Generate(TinyGPT model, Tensor idx, int maxNewTokens, int topK = 0, double temperature = 1.0)
        {
            var device = idx.device;
            var B = idx.shape[0];
            var T0 = idx.shape[1];

            var cur = idx.clone();

            for (int step = 0; step < maxNewTokens; step++)
            {
                // If sequence longer than model's maxT, crop from the right
                long maxT = (model as dynamic).GetType().GetField("maxT",
                    System.Reflection.BindingFlags.NonPublic |
                    System.Reflection.BindingFlags.Instance) is var _ ? 0 : 0; // not exposed; we just crop to last 128 by default
                long ctxLen = 128; // use your TinyGPT maxSeqLen; keep in sync with constructor
                var ctx = cur.shape[1] > ctxLen
                    ? cur.index(new TensorIndex[] { TensorIndex.Ellipsis, TensorIndex.Slice(-(long)ctxLen, null) })
                    : cur;

                var logits = model.forward(ctx); // [B, Tctx, V]
                var last = logits.index(new TensorIndex[] { TensorIndex.Ellipsis, ctx.shape[1] - 1, TensorIndex.Ellipsis }); // [B, V]

                // Temperature
                if (temperature != 1.0)
                    last = last / temperature;

                // Top-k filter
                if (topK > 0 && topK < last.shape[last.ndim - 1])
                {
                    var (vals, inds) = last.topk(topK, dim: -1);
                    var minVals = vals.index(new TensorIndex[] { TensorIndex.Ellipsis, topK - 1 }).unsqueeze(-1);
                    var mask = last.lt(minVals); // logits < kth largest -> mask
                    var negInf = torch.full_like(last, double.NegativeInfinity);
                    last = torch.where(mask, negInf, last);
                }

                // Sample from distribution
                var probs = functional.softmax(last, dim: -1);
                var next = probs.multinomial(1); // [B,1]

                // Append
                cur = torch.cat(new[] { cur, next }, dim: 1);
            }

            return cur;
        }
    }
}

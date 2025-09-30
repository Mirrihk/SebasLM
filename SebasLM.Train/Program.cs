// SebasLM.Train/Program.cs
// Build/Run: dotnet run -c Release --project SebasLM.Train

using System;
using System.Linq;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using SebasLM.Model; // <- from SebasLM.Core

namespace SebasLM.Train
{
    static class DeviceUtil
    {
        public static Device Best => cuda.is_available() ? CUDA : CPU;
        private static Device CUDA => torch.CUDA;
        private static Device CPU => torch.CPU;
    }

    // Simple printable-ASCII tokenizer (32..126)
    sealed class TinyCharTokenizer
    {
        public int VocabSize => 95;
        private readonly char[] idx2ch = Enumerable.Range(32, 95).Select(i => (char)i).ToArray();
        private readonly System.Collections.Generic.Dictionary<char, int> ch2idx;

        public TinyCharTokenizer()
        {
            ch2idx = idx2ch.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);
        }

        public Tensor Encode(string s, Device device)
        {
            var ids = s.Select(c => ch2idx.TryGetValue(c, out var id) ? id : 0).ToArray();
            return torch.tensor(ids, dtype: ScalarType.Int64, device: device);
        }

        public string Decode(Tensor ids)
        {
            var data = ids.to_type(ScalarType.Int64).cpu().data<long>().ToArray();
            var sb = new StringBuilder();
            foreach (var id in data) sb.Append(idx2ch[(int)Math.Clamp(id, 0, idx2ch.Length - 1)]);
            return sb.ToString();
        }
    }

    class Program
    {
        static void Main()
        {
            var device = DeviceUtil.Best;
            Console.WriteLine($"Device: {device.type}");

            var tok = new TinyCharTokenizer();

            // Put your own text here to personalize the demo
            var corpus = "hello from sebaslm — a tiny transformer in c# with torchsharp!\n";
            const int blockSize = 128;
            const int batchSize = 16;

            // Build the model from SebasLM.Core (you already added these classes)
            var model = new MiniTransformer(
                name: "sebas_minilm",
                vocabSize: tok.VocabSize,
                dModel: 256, nLayers: 4, nHeads: 8, dHidden: 768,
                maxSeqLen: blockSize, dropoutP: 0.0
            ).to(device);

            var optim = torch.optim.AdamW(model.parameters(), lr: 3e-4, betas: (0.9, 0.95), weight_decay: 0.01);

            // Encode data and (if too short) repeat once so we can sample batches
            var data = tok.Encode(corpus, device);
            if (data.shape[0] < blockSize + 1)
                data = torch.cat(new Tensor[] { data, data }, 0);

            // Tiny training loop
            int steps = 200;
            model.train();
            for (int step = 1; step <= steps; step++)
            {
                var (x, y) = SampleBatch(data, batchSize, blockSize, device);
                optim.zero_grad();

                var logits = model.forward(x); // [B,T,V]
                var loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1)
                );

                loss.backward();
                optim.step();

                if (step % 25 == 0)
                    Console.WriteLine($"step {step}/{steps} | loss {loss.ToSingle():0.000}");
            }

            // Generate a sample
            model.eval();
            var prompt = "hello";
            var start = tok.Encode(prompt, device).unsqueeze(0); // [1,T]
            var gen = model.Generate(start, maxNewTokens: 120, topK: 30, temperature: 0.9);

            Console.WriteLine("\n=== Generated ===");
            Console.WriteLine(tok.Decode(gen[0]));
        }

        // Batch sampler using TorchSharp-friendly slicing
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
    }
}

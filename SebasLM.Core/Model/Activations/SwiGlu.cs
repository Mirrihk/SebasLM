using System;
using System.Runtime.CompilerServices;

namespace SebasLM.Core.Model.Activations
{
    /// <summary>
    /// SwiGLU (Swish-Gated Linear Unit) activation used in modern Transformer MLPs.
    /// 
    /// Given an input X shaped [batch, 2d], split the last dimension into A and B (each [batch, d]),
    /// then compute Y = SiLU(A) ⊙ B with SiLU(x) = x * sigmoid(x). Y has shape [batch, d].
    /// 
    /// Typical usage inside an FFN block:
    ///   h = Linear1(x)              // d -> 2d
    ///   y = SwiGLU.Forward(h, d)    // [batch, 2d] -> [batch, d]
    ///   out = Linear2(y)            // d -> d
    /// 
    /// This implementation operates on flattened buffers for CPU-friendly performance.
    /// </summary>
    public static class SwiGLU
    {
        // ---------------------------------------------------------------------
        // Forward pass
        // ---------------------------------------------------------------------

        /// <summary>
        /// Forward for flattened buffers.
        /// X length must be batch * (2 * d). Y length must be batch * d.
        /// Splits X into A (first d) and B (second d) per batch row, then writes Y = SiLU(A) * B.
        /// </summary>
        /// <param name="x">Read-only span over the input buffer [batch, 2d].</param>
        /// <param name="y">Output span for the result [batch, d].</param>
        /// <param name="d">Hidden size per half (so X has last-dim 2*d).</param>
        public static void Forward(ReadOnlySpan<float> x, Span<float> y, int d)
        {
            if (x.IsEmpty) throw new ArgumentException("Input x is empty.", nameof(x));
            if (y.IsEmpty) throw new ArgumentException("Output y is empty.", nameof(y));
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(d), "d must be > 0.");

            int twoD = 2 * d;
            if (x.Length % twoD != 0)
                throw new ArgumentException($"Input length {x.Length} is not a multiple of 2*d ({twoD}).", nameof(x));

            int batch = x.Length / twoD;
            if (y.Length != batch * d)
                throw new ArgumentException($"Output length {y.Length} must equal batch*d ({batch * d}).", nameof(y));

            for (int b = 0; b < batch; b++)
            {
                int xBase = b * twoD;
                int yBase = b * d;

                for (int i = 0; i < d; i++)
                {
                    float a  = x[xBase + i];       // A: first half
                    float bb = x[xBase + d + i];   // B: second half

                    float s = Sigmoid(a);
                    float silu = a * s;            // SiLU(a)
                    y[yBase + i] = silu * bb;      // SwiGLU = SiLU(A) * B
                }
            }
        }

        // ---------------------------------------------------------------------
        // Backward pass
        // ---------------------------------------------------------------------

        /// <summary>
        /// Backward for flattened buffers.
        /// Given input X ([batch, 2d]) and upstream gradient dY ([batch, d]),
        /// writes dX ([batch, 2d]) where the first d are grads for A and the second d for B.
        /// 
        /// Formulas:
        ///   y = SiLU(a) * b, where SiLU(a) = a * σ(a)
        ///   dSiLU/da = σ(a) * (1 + a * (1 - σ(a)))
        ///   ∂y/∂a = b * dSiLU/da
        ///   ∂y/∂b = SiLU(a)
        ///   => grad_a = g * b * dSiLU/da
        ///      grad_b = g * SiLU(a)
        /// </summary>
        /// <param name="x">Read-only span over the input buffer [batch, 2d].</param>
        /// <param name="dy">Read-only span over upstream gradient [batch, d].</param>
        /// <param name="dx">Output span for input gradient [batch, 2d].</param>
        /// <param name="d">Hidden size per half.</param>
        public static void Backward(ReadOnlySpan<float> x, ReadOnlySpan<float> dy, Span<float> dx, int d)
        {
            if (x.IsEmpty)  throw new ArgumentException("Input x is empty.", nameof(x));
            if (dy.IsEmpty) throw new ArgumentException("Input dy is empty.", nameof(dy));
            if (dx.IsEmpty) throw new ArgumentException("Output dx is empty.", nameof(dx));
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(d), "d must be > 0.");

            int twoD = 2 * d;
            if (x.Length % twoD != 0)
                throw new ArgumentException($"Input length {x.Length} is not a multiple of 2*d ({twoD}).", nameof(x));

            int batch = x.Length / twoD;
            if (dy.Length != batch * d)
                throw new ArgumentException($"dy length {dy.Length} must equal batch*d ({batch * d}).", nameof(dy));

            if (dx.Length != batch * twoD)
                throw new ArgumentException($"dx length {dx.Length} must equal batch*2*d ({batch * twoD}).", nameof(dx));

            for (int b = 0; b < batch; b++)
            {
                int xBase = b * twoD;
                int yBase = b * d;

                for (int i = 0; i < d; i++)
                {
                    float a  = x[xBase + i];
                    float bb = x[xBase + d + i];
                    float g  = dy[yBase + i];

                    float s = Sigmoid(a);
                    float silu  = a * s;
                    float dSiLU = s * (1f + a * (1f - s));  // d/da SiLU(a)

                    float grad_a = g * bb * dSiLU;          // ∂L/∂a
                    float grad_b = g * silu;                // ∂L/∂b

                    dx[xBase + i]       = grad_a;           // write grads for A
                    dx[xBase + d + i]   = grad_b;           // write grads for B
                }
            }
        }

        // ---------------------------------------------------------------------
        // Convenience overloads
        // ---------------------------------------------------------------------

        /// <summary>
        /// Forward overload when you already know (batch, d).
        /// X is [batch, 2d], Y is [batch, d].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Forward(int batch, int d, ReadOnlySpan<float> x, Span<float> y)
            => Forward(x, y, d);

        /// <summary>
        /// Backward overload when you already know (batch, d).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Backward(int batch, int d, ReadOnlySpan<float> x, ReadOnlySpan<float> dy, Span<float> dx)
            => Backward(x, dy, dx, d);

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sigmoid(float v)
        {
            // Fast, stable enough for typical ranges; clip or branch if you expect extreme values.
            return 1f / (1f + MathF.Exp(-v));
        }
    }
}

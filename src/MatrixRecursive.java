import java.util.Arrays;
import java.util.Locale;

public final class MatrixRecursive {
    public static final String IMPLEMENTATION = "Schur + block LU, no pivoting";
    public enum Method { BINET, STRASSEN, LADERMAN }

    private MatrixRecursive() {
    }

    public static final class Stats {
        // Wszystkie +, -, *, / wykonane WEWNATRZ mnozen macierzy.
        public final long multiplicationFlops;
        // Operacje poza mnozeniami: odejmowanie blokow, dzielenie itd.
        public final long otherFlops;

        private Stats(long multiplicationFlops, long otherFlops) {
            this.multiplicationFlops = multiplicationFlops;
            this.otherFlops = otherFlops;
        }

        public long totalFlops() {
            return multiplicationFlops + otherFlops;
        }
    }

    public static final class LUResult {
        public final double[][] L, U;
        // Dla zgodnosci interfejsu: permutacja identycznosciowa.
        // Ten wariant nie zamienia wierszy: A = L*U.
        public final int[] permutation;
        public final Stats stats;

        private LUResult(double[][] l, double[][] u, int[] p, Stats stats) {
            this.L = l;
            this.U = u;
            this.permutation = p;
            this.stats = stats;
        }
    }

    public static final class InverseResult {
        public final double[][] inverse;
        public final Stats stats;

        private InverseResult(double[][] inverse, Stats stats) {
            this.inverse = inverse;
            this.stats = stats;
        }
    }

    public static LUResult lu(double[][] a, Method method) {
        checkSquare(a);
        return new Calculation(method).decompose(a);
    }

    public static InverseResult inverse(double[][] a, Method method) {
        checkSquare(a);
        Calculation calculation = new Calculation(method);
        double[][] result = calculation.inverseRecursive(a);
        checkFinite(result);
        return new InverseResult(result, calculation.stats());
    }

    // Osobny obiekt i osobne liczniki dla kazdego wywolania lu/inverse.
    private static final class Calculation {
        private final Method method;
        private long multiplicationFlops;
        private long otherFlops;

        private Calculation(Method method) {
            if (method == null)
                throw new IllegalArgumentException("Wybierz metode mnozenia.");
            this.method = method;
        }

        private Stats stats() {
            return new Stats(multiplicationFlops, otherFlops);
        }

        private double[][] multiply(double[][] a, double[][] b) {
            long before = MatrixMultAlt.flops;
            double[][] result;
            switch (method) {
                case BINET: result = MatrixMultAlt.binet(a, b); break;
                case STRASSEN: result = MatrixMultAlt.strassen(a, b); break;
                case LADERMAN: result = MatrixMultAlt.laderman(a, b); break;
                default: throw new AssertionError(method);
            }
            multiplicationFlops += MatrixMultAlt.flops - before;
            return result;
        }

        private double[][] subtract(double[][] a, double[][] b) {
            int rows = a.length, cols = a[0].length;
            double[][] c = new double[rows][cols];
            otherFlops += (long) rows * cols;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    c[i][j] = a[i][j] - b[i][j];
            return c;
        }

        private double[][] negate(double[][] a) {
            int rows = a.length, cols = a[0].length;
            double[][] c = new double[rows][cols];
            otherFlops += (long) rows * cols;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    c[i][j] = 0.0 - a[i][j];
            return c;
        }

        private double[][] add(double[][] a, double[][] b) {
            int rows = a.length, cols = a[0].length;
            double[][] c = new double[rows][cols];
            otherFlops += (long) rows * cols;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    c[i][j] = a[i][j] + b[i][j];
            return c;
        }

        // Bezposrednie odwracanie blokowe przez dopelnienie Schura.
        // Wymaga odwracalnosci wszystkich blokow odwracanych w rekurencji.
        private double[][] inverseRecursive(double[][] a) {
            int n = a.length;
            if (n == 1) {
                requireNonzeroPivot(a[0][0]);
                otherFlops++;
                return new double[][]{{1.0 / a[0][0]}};
            }

            int k = n / 2, rest = n - k;
            double[][] a11 = submatrix(a, 0, 0, k, k);
            double[][] a12 = submatrix(a, 0, k, k, rest);
            double[][] a21 = submatrix(a, k, 0, rest, k);
            double[][] a22 = submatrix(a, k, k, rest, rest);

            double[][] d = inverseRecursive(a11);
            double[][] x = multiply(d, a12);       // X = A11^(-1) * A12.
            double[][] y = multiply(a21, d);       // Y = A21 * A11^(-1).
            double[][] s = subtract(a22, multiply(y, a12));
            double[][] z = inverseRecursive(s);    // Z = S^(-1).

            double[][] xz = multiply(x, z);
            double[][] b11 = add(d, multiply(xz, y));
            double[][] b12 = negate(xz);           // Bez odwracania A12!
            double[][] b21 = negate(multiply(z, y));
            return join(b11, b12, b21, z);
        }

        private LUResult decompose(double[][] a) {
            double[][][] factors = luRecursive(a);
            checkFinite(factors[0]);
            checkFinite(factors[1]);
            int[] permutation = new int[a.length];
            for (int i = 0; i < a.length; i++) permutation[i] = i;
            return new LUResult(factors[0], factors[1], permutation, stats());
        }

        // Rekurencyjne LU wedlug slajdu, bez zamiany wierszy.
        // Wynik [0] to L, wynik [1] to U.
        private double[][][] luRecursive(double[][] a) {
            int n = a.length;
            if (n == 1) {
                requireNonzeroPivot(a[0][0]);
                return new double[][][]{ {{1.0}}, {{a[0][0]}} };
            }

            int k = n / 2, rest = n - k;
            double[][] a11 = submatrix(a, 0, 0, k, k);
            double[][] a12 = submatrix(a, 0, k, k, rest);
            double[][] a21 = submatrix(a, k, 0, rest, k);
            double[][] a22 = submatrix(a, k, k, rest, rest);

            double[][][] first = luRecursive(a11);
            double[][] l11 = first[0], u11 = first[1];

            double[][] u11Inverse = inverseRecursive(u11);
            double[][] l21 = multiply(a21, u11Inverse);
            double[][] l11Inverse = inverseRecursive(l11);
            double[][] u12 = multiply(l11Inverse, a12);

            double[][] s = subtract(a22, multiply(l21, u12));
            double[][][] second = luRecursive(s);
            double[][] l22 = second[0], u22 = second[1];

            // Zera sa czescia struktury macierzy trojkatnych, nie paddingiem.
            double[][] l = join(l11, new double[k][rest], l21, l22);
            double[][] u = join(u11, u12, new double[rest][k], u22);
            return new double[][][]{l, u};
        }
    }

    private static void requireNonzeroPivot(double pivot) {
        if (pivot == 0.0)
            throw new ArithmeticException("Zerowy pivot w bloku: algorytm bez "
                    + "zamiany wierszy nie moze kontynuowac dla tej macierzy.");
    }

    private static double[][] join(double[][] a11, double[][] a12,
                                   double[][] a21, double[][] a22) {
        int top = a11.length, left = a11[0].length;
        double[][] result = new double[top + a21.length][left + a12[0].length];
        put(result, 0, 0, a11);
        put(result, 0, left, a12);
        put(result, top, 0, a21);
        put(result, top, left, a22);
        return result;
    }

    private static double[][] submatrix(double[][] a, int row, int col,
                                        int rows, int cols) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(a[row + i], col, result[i], 0, cols);
        return result;
    }

    private static void put(double[][] target, int row, int col, double[][] block) {
        for (int i = 0; i < block.length; i++)
            System.arraycopy(block[i], 0, target[row + i], col, block[i].length);
    }

    private static void checkSquare(double[][] a) {
        if (a == null || a.length == 0)
            throw new IllegalArgumentException("Macierz nie moze byc pusta.");
        for (double[] row : a) {
            if (row == null || row.length != a.length)
                throw new IllegalArgumentException("Macierz musi byc kwadratowa.");
            for (double value : row)
                if (!Double.isFinite(value))
                    throw new IllegalArgumentException("Elementy musza byc skonczone.");
        }
    }

    private static void checkFinite(double[][] a) {
        for (double[] row : a)
            for (double value : row)
                if (!Double.isFinite(value))
                    throw new ArithmeticException("Przepelnienie lub niepoprawny wynik double.");
    }

    // Weryfikacja poza licznikami algorytmow.
    private static double productError(double[][] a, double[][] b, double[][] target) {
        double error = 0.0;
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b[0].length; j++) {
                double value = 0.0;
                for (int k = 0; k < b.length; k++) value += a[i][k] * b[k][j];
                error = Math.max(error, Math.abs(value - target[i][j]));
            }
        return error;
    }

    private static void print(String name, double[][] a) {
        System.out.println(name + ":");
        for (double[] row : a) {
            for (double value : row) System.out.printf(Locale.ROOT, "%12.6f", value);
            System.out.println();
        }
    }

    private static void printStats(Stats stats) {
        System.out.println("FLOP w mnozeniach: " + stats.multiplicationFlops);
        System.out.println("FLOP poza mnozeniami: " + stats.otherFlops);
        System.out.println("FLOP razem: " + stats.totalFlops());
    }

    public static void main(String[] args) {
        double[][] a = {
                {10, 2, 1, 0, 1},
                {1, 11, 1, 1, 0},
                {1, 0, 12, 1, 1},
                {1, 1, 0, 13, 1},
                {0, 1, 1, 1, 14}
        };
        int n = a.length;
        double[][] identity = new double[n][n];
        for (int i = 0; i < n; i++) identity[i][i] = 1.0;

        print("A", a);
        for (Method method : Method.values()) {
            System.out.println("\nODWRACANIE - " + method);
            InverseResult result = inverse(a, method);
            print("A^(-1)", result.inverse);
            printStats(result.stats);
            System.out.println("max |A*A^(-1) - I| = "
                    + productError(a, result.inverse, identity));

            // Zadanie wymaga prezentacji LU dla Bineta oraz Strassena.
            if (method != Method.LADERMAN) {
                System.out.println("\nLU - " + method);
                LUResult lu = lu(a, method);
                print("L", lu.L);
                print("U", lu.U);
                System.out.println("Permutacja wierszy: " + Arrays.toString(lu.permutation));
                printStats(lu.stats);

                double[][] pa = new double[n][n];
                for (int i = 0; i < n; i++) pa[i] = a[lu.permutation[i]].clone();
                System.out.println("max |L*U - A| = " + productError(lu.L, lu.U, pa));
            }
        }
    }
}
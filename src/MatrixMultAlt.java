import java.util.Random;

public final class MatrixMultAlt {
    // Dla n <= CUTOFF liczymy zwyklymi petlami.
    private static final int CUTOFF = 2;

    private MatrixMultAlt() {
    }

    //  Klasyczne mnożenie macierzy o dowolnych rozmiarach
    public static double[][] classical(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;
        double[][] c = new double[m][p];
        multiplySubmatrix(a, 0, 0, b, 0, 0, c, 0, 0, m, k, p);
        return c;
    }

    //  Mnożenie danych podmacierzy
    private static void multiplySubmatrix(double[][] a, int ar, int ac,
                                    double[][] b, int br, int bc,
                                    double[][] c, int cr, int cc,
                                    int m, int k, int p) {
        for (int i = 0; i < m; i++)
            for (int t = 0; t < k; t++)
                for (int j = 0; j < p; j++)
                    c[cr + i][cc + j] += a[ar + i][ac + t] * b[br + t][bc + j];
    }

    public static double[][] binet(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;
        double[][] c = new double[m][p];
        binetAdd(a, 0, 0, b, 0, 0, c, 0, 0, m, k, p);
        return c;
    }

    private static void binetAdd(double[][] a, int ar, int ac,
                                 double[][] b, int br, int bc,
                                 double[][] c, int cr, int cc,
                                 int m, int k, int p) {
        if (Math.min(m, Math.min(k, p)) <= CUTOFF) {
            multiplySubmatrix(a, ar, ac, b, br, bc, c, cr, cc, m, k, p);
            return;
        }
        // Punkty podzialu wzdluz kazdego wymiaru: {0, polowa, koniec}.
        int[] mi = {0, m / 2, m}, ki = {0, k / 2, k}, pi = {0, p / 2, p};
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int t = 0; t < 2; t++)
                    // C[i][j] += A[i][t] * B[t][j]
                    binetAdd(a, ar + mi[i], ac + ki[t],
                            b, br + ki[t], bc + pi[j],
                            c, cr + mi[i], cc + pi[j],
                            mi[i + 1] - mi[i], ki[t + 1] - ki[t], pi[j + 1] - pi[j]);
    }

    private static double[][][] split(double[][] a, int parts) {
        int h = a.length / parts, w = a[0].length / parts;
        double[][][] blocks = new double[parts * parts + 1][][];
        for (int i = 0; i < parts; i++)
            for (int j = 0; j < parts; j++) {
                double[][] blk = new double[h][w];
                for (int r = 0; r < h; r++)
                    System.arraycopy(a[i * h + r], j * w, blk[r], 0, w);
                blocks[1 + i * parts + j] = blk;
            }
        return blocks;
    }

    /** Odwrotnosc split: sklada bloki 1..parts*parts w jedna macierz. */
    private static double[][] join(double[][][] blocks, int parts) {
        int h = blocks[1].length, w = blocks[1][0].length;
        double[][] a = new double[parts * h][parts * w];
        for (int i = 0; i < parts; i++)
            for (int j = 0; j < parts; j++)
                for (int r = 0; r < h; r++)
                    System.arraycopy(blocks[1 + i * parts + j][r], 0, a[i * h + r], j * w, w);
        return a;
    }

    /** Kombinacja liniowa blokow o tym samym ksztalcie. Liczba w terms to numer bloku,
     * a jej znak to znak skladnika: lin(x, 1, -4) = x[1] - x[4]. */
    private static double[][] lin(double[][][] blocks, int... terms) {
        int h = blocks[Math.abs(terms[0])].length, w = blocks[Math.abs(terms[0])][0].length;
        double[][] c = new double[h][w];
        for (int term : terms) {
            double[][] b = blocks[Math.abs(term)];
            double sign = term > 0 ? 1 : -1;
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                    c[i][j] += sign * b[i][j];
        }
        return c;
    }

    /** Dopelnia macierz zerami do rozmiaru rows x cols. */
    private static double[][] pad(double[][] a, int rows, int cols) {
        double[][] b = new double[rows][cols];
        for (int i = 0; i < a.length; i++)
            System.arraycopy(a[i], 0, b[i], 0, a[0].length);
        return b;
    }

    /** Zwraca lewy gorny rog rows x cols. */
    private static double[][] crop(double[][] a, int rows, int cols) {
        double[][] b = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(a[i], 0, b[i], 0, cols);
        return b;
    }

    /** Najmniejsza wielokrotnosc parts nie mniejsza niz x. */
    private static int roundUp(int x, int parts) {
        return (x + parts - 1) / parts * parts;
    }

    // ===== Strassen: 7 mnozen zamiast 8 (A: m x k, B: k x p) =====

    public static double[][] strassen(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;
        if (Math.min(m, Math.min(k, p)) <= CUTOFF) return classical(a, b);
        int m2 = roundUp(m, 2), k2 = roundUp(k, 2), p2 = roundUp(p, 2);
        if (m2 != m || k2 != k || p2 != p)
            return crop(strassen(pad(a, m2, k2), pad(b, k2, p2)), m, p);

        double[][][] x = split(a, 2), y = split(b, 2); // 1=11, 2=12, 3=21, 4=22
        double[][][] m_ = new double[8][][];
        m_[1] = strassen(lin(x, 1, 4), lin(y, 1, 4));
        m_[2] = strassen(lin(x, 3, 4), y[1]);
        m_[3] = strassen(x[1], lin(y, 2, -4));
        m_[4] = strassen(x[4], lin(y, 3, -1));
        m_[5] = strassen(lin(x, 1, 2), y[4]);
        m_[6] = strassen(lin(x, 3, -1), lin(y, 1, 2));
        m_[7] = strassen(lin(x, 2, -4), lin(y, 3, 4));

        double[][][] c = new double[5][][];
        c[1] = lin(m_, 1, 4, -5, 7);  // C11 = M1 + M4 - M5 + M7
        c[2] = lin(m_, 3, 5);         // C12 = M3 + M5
        c[3] = lin(m_, 2, 4);         // C21 = M2 + M4
        c[4] = lin(m_, 1, -2, 3, 6);  // C22 = M1 - M2 + M3 + M6
        return join(c, 2);
    }

    // ===== Laderman: 23 mnozenia zamiast 27 (podzial 3 x 3) =====
    // Wzory: J. D. Laderman, Bull. AMS 82 (1976), 126-128.

    public static double[][] laderman(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;
        if (Math.min(m, Math.min(k, p)) <= CUTOFF) return classical(a, b);
        int m3 = roundUp(m, 3), k3 = roundUp(k, 3), p3 = roundUp(p, 3);
        if (m3 != m || k3 != k || p3 != p)
            return crop(laderman(pad(a, m3, k3), pad(b, k3, p3)), m, p);

        double[][][] x = split(a, 3), y = split(b, 3); // 1..9 = 11,12,13,21,22,23,31,32,33
        double[][][] m_ = new double[24][][];
        m_[1] = laderman(lin(x, 1, 2, 3, -4, -5, -8, -9), y[5]);
        m_[2] = laderman(lin(x, 1, -4), lin(y, -2, 5));
        m_[3] = laderman(x[5], lin(y, -1, 2, 4, -5, -6, -7, 9));
        m_[4] = laderman(lin(x, -1, 4, 5), lin(y, 1, -2, 5));
        m_[5] = laderman(lin(x, 4, 5), lin(y, -1, 2));
        m_[6] = laderman(x[1], y[1]);
        m_[7] = laderman(lin(x, -1, 7, 8), lin(y, 1, -3, 6));
        m_[8] = laderman(lin(x, -1, 7), lin(y, 3, -6));
        m_[9] = laderman(lin(x, 7, 8), lin(y, -1, 3));
        m_[10] = laderman(lin(x, 1, 2, 3, -5, -6, -7, -8), y[6]);
        m_[11] = laderman(x[8], lin(y, -1, 3, 4, -5, -6, -7, 8));
        m_[12] = laderman(lin(x, -3, 8, 9), lin(y, 5, 7, -8));
        m_[13] = laderman(lin(x, 3, -9), lin(y, 5, -8));
        m_[14] = laderman(x[3], y[7]);
        m_[15] = laderman(lin(x, 8, 9), lin(y, -7, 8));
        m_[16] = laderman(lin(x, -3, 5, 6), lin(y, 6, 7, -9));
        m_[17] = laderman(lin(x, 3, -6), lin(y, 6, -9));
        m_[18] = laderman(lin(x, 5, 6), lin(y, -7, 9));
        m_[19] = laderman(x[2], y[4]);
        m_[20] = laderman(x[6], y[8]);
        m_[21] = laderman(x[4], y[3]);
        m_[22] = laderman(x[7], y[2]);
        m_[23] = laderman(x[9], y[9]);

        double[][][] c = new double[10][][];
        c[1] = lin(m_, 6, 14, 19);
        c[2] = lin(m_, 1, 4, 5, 6, 12, 14, 15);
        c[3] = lin(m_, 6, 7, 9, 10, 14, 16, 18);
        c[4] = lin(m_, 2, 3, 4, 6, 14, 16, 17);
        c[5] = lin(m_, 2, 4, 5, 6, 20);
        c[6] = lin(m_, 14, 16, 17, 18, 21);
        c[7] = lin(m_, 6, 7, 8, 11, 12, 13, 14);
        c[8] = lin(m_, 12, 13, 14, 15, 22);
        c[9] = lin(m_, 6, 7, 8, 9, 23);
        return join(c, 3);
    }

    private static void print(String nazwa, double[][] x) {
        System.out.println(nazwa + " (" + x.length + " x " + x[0].length + "):");
        for (double[] row : x) {
            for (double v : row) System.out.printf("%6.0f", v);
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Random r = new Random(1);
        double[][] a = new double[5][4], b = new double[4][6];
        for (double[] row : a) for (int j = 0; j < row.length; j++) row[j] = r.nextInt(10);
        for (double[] row : b) for (int j = 0; j < row.length; j++) row[j] = r.nextInt(10);

        double[][] c1 = classical(a, b), c2 = binet(a, b);
        print("Classical", c1);
        print("Binet", c2);
        print("Strassen", strassen(a, b));
        print("Laderman", laderman(a, b));
    }

}
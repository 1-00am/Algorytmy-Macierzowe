import java.util.Random;

public final class MatrixMultAlt {
    // Dla najmniejszego wymiaru <= CUTOFF liczymy zwyklymi petlami.
    static int CUTOFF = 2;
    static long flops;

    private MatrixMultAlt() {
    }

    // Zakladamy niepuste, prostokatne macierze A: m x k oraz B: k x p.
    public static double[][] classical(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;
        double[][] c = new double[m][p];
        multiplySubmatrix(a, 0, 0, b, 0, 0, c, 0, 0, m, k, p);
        return c;
    }

    // C += A * B dla wskazanych podmacierzy. Wynik jest DODAWANY do C.
    private static void multiplySubmatrix(double[][] a, int ar, int ac,
                                          double[][] b, int br, int bc,
                                          double[][] c, int cr, int cc,
                                          int m, int k, int p) {
        flops += 2L * m * k * p;
        for (int i = 0; i < m; i++)
            for (int t = 0; t < k; t++)
                for (int j = 0; j < p; j++)
                    c[cr + i][cc + j] +=
                            a[ar + i][ac + t] * b[br + t][bc + j];
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
        if (Math.min(m, Math.min(k, p)) <= Math.max(CUTOFF, 1)) {
            multiplySubmatrix(a, ar, ac, b, br, bc, c, cr, cc, m, k, p);
            return;
        }

        // Bloki moga miec rozne rozmiary, np. 5 dzielimy na 2 + 3.
        int[] mi = {0, m / 2, m};
        int[] ki = {0, k / 2, k};
        int[] pi = {0, p / 2, p};

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int t = 0; t < 2; t++)
                    binetAdd(a, ar + mi[i], ac + ki[t],
                            b, br + ki[t], bc + pi[j],
                            c, cr + mi[i], cc + pi[j],
                            mi[i + 1] - mi[i],
                            ki[t + 1] - ki[t],
                            pi[j + 1] - pi[j]);
    }

    // Kopia istniejacego fragmentu, bez powiekszania i dopisywania zer.
    private static double[][] submatrix(double[][] a, int row, int col,
                                        int rows, int cols) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(a[row + i], col, result[i], 0, cols);
        return result;
    }

    // Wywolywane tylko wtedy, gdy oba wymiary sa podzielne przez parts.
    private static double[][][] split(double[][] a, int parts) {
        int h = a.length / parts, w = a[0].length / parts;
        double[][][] blocks = new double[parts * parts + 1][][];

        for (int i = 0; i < parts; i++)
            for (int j = 0; j < parts; j++)
                blocks[1 + i * parts + j] =
                        submatrix(a, i * h, j * w, h, w);

        return blocks;
    }

    private static double[][] join(double[][][] blocks, int parts) {
        int h = blocks[1].length, w = blocks[1][0].length;
        double[][] a = new double[parts * h][parts * w];

        for (int i = 0; i < parts; i++)
            for (int j = 0; j < parts; j++)
                for (int r = 0; r < h; r++)
                    System.arraycopy(blocks[1 + i * parts + j][r], 0,
                            a[i * h + r], j * w, w);

        return a;
    }

    private static double[][] lin(double[][][] blocks, int... terms) {
        int h = blocks[Math.abs(terms[0])].length;
        int w = blocks[Math.abs(terms[0])][0].length;

        double[][] c = new double[h][w];
        flops += (long) terms.length * h * w;

        for (int term : terms) {
            double[][] b = blocks[Math.abs(term)];

            if (term > 0) {
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        c[i][j] += b[i][j];
            } else {
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        c[i][j] -= b[i][j];
            }
        }

        return c;
    }

    // Obsluga reszty z dzielenia wymiarow przez 2 albo 3, BEZ paddingu.
    private static double[][] multiplyWithRemainder(double[][] a,
                                                    double[][] b,
                                                    int parts) {
        int m = a.length, k = b.length, p = b[0].length;

        int m0 = m - m % parts;
        int k0 = k - k % parts;
        int p0 = p - p % parts;

        // Wymiary czesci glownej sa podzielne przez parts.
        // Przypadki z wymiarem < parts obsluguje warunek bazowy rekurencji.
        double[][] a0 = submatrix(a, 0, 0, m0, k0);
        double[][] b0 = submatrix(b, 0, 0, k0, p0);

        double[][] c0 = parts == 2
                ? strassen(a0, b0)
                : laderman(a0, b0);

        double[][] c = new double[m][p];

        for (int i = 0; i < m0; i++)
            System.arraycopy(c0[i], 0, c[i], 0, p0);

        // 1. Brakujacy wklad do lewego gornego bloku:
        //    ostatnie kolumny A * ostatnie wiersze B.
        if (k0 < k)
            multiplySubmatrix(a, 0, k0,
                    b, k0, 0,
                    c, 0, 0,
                    m0, k - k0, p0);

        // 2. Prawe kolumny wyniku, tylko w gornych m0 wierszach.
        if (p0 < p)
            multiplySubmatrix(a, 0, 0,
                    b, 0, p0,
                    c, 0, p0,
                    m0, k, p - p0);

        // 3. Dolne wiersze wyniku, na calej szerokosci.
        //    Prawy dolny rog jest liczony tylko tutaj.
        if (m0 < m)
            multiplySubmatrix(a, m0, 0,
                    b, 0, 0,
                    c, m0, 0,
                    m - m0, k, p);

        return c;
    }

    // Strassen: 7 mnozen blokowych zamiast 8.
    public static double[][] strassen(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;

        if (Math.min(m, Math.min(k, p)) <= Math.max(CUTOFF, 1))
            return classical(a, b);

        if (m % 2 != 0 || k % 2 != 0 || p % 2 != 0)
            return multiplyWithRemainder(a, b, 2);

        double[][][] x = split(a, 2), y = split(b, 2);
        double[][][] mm = new double[8][][];

        mm[1] = strassen(lin(x, 1, 4), lin(y, 1, 4));
        mm[2] = strassen(lin(x, 3, 4), y[1]);
        mm[3] = strassen(x[1], lin(y, 2, -4));
        mm[4] = strassen(x[4], lin(y, 3, -1));
        mm[5] = strassen(lin(x, 1, 2), y[4]);
        mm[6] = strassen(lin(x, 3, -1), lin(y, 1, 2));
        mm[7] = strassen(lin(x, 2, -4), lin(y, 3, 4));

        double[][][] c = new double[5][][];
        c[1] = lin(mm, 1, 4, -5, 7);
        c[2] = lin(mm, 3, 5);
        c[3] = lin(mm, 2, 4);
        c[4] = lin(mm, 1, -2, 3, 6);

        return join(c, 2);
    }

    // Laderman: 23 mnozenia blokowe zamiast 27 (podzial 3 x 3).
    public static double[][] laderman(double[][] a, double[][] b) {
        int m = a.length, k = b.length, p = b[0].length;

        // Macierzy z wymiarem 1 albo 2 nie dzielimy na 3 czesci.
        if (Math.min(m, Math.min(k, p)) <= Math.max(CUTOFF, 2))
            return classical(a, b);

        if (m % 3 != 0 || k % 3 != 0 || p % 3 != 0)
            return multiplyWithRemainder(a, b, 3);

        double[][][] x = split(a, 3), y = split(b, 3);
        double[][][] mm = new double[24][][];

        mm[1] = laderman(lin(x, 1, 2, 3, -4, -5, -8, -9), y[5]);
        mm[2] = laderman(lin(x, 1, -4), lin(y, -2, 5));
        mm[3] = laderman(x[5], lin(y, -1, 2, 4, -5, -6, -7, 9));
        mm[4] = laderman(lin(x, -1, 4, 5), lin(y, 1, -2, 5));
        mm[5] = laderman(lin(x, 4, 5), lin(y, -1, 2));
        mm[6] = laderman(x[1], y[1]);
        mm[7] = laderman(lin(x, -1, 7, 8), lin(y, 1, -3, 6));
        mm[8] = laderman(lin(x, -1, 7), lin(y, 3, -6));
        mm[9] = laderman(lin(x, 7, 8), lin(y, -1, 3));
        mm[10] = laderman(lin(x, 1, 2, 3, -5, -6, -7, -8), y[6]);
        mm[11] = laderman(x[8], lin(y, -1, 3, 4, -5, -6, -7, 8));
        mm[12] = laderman(lin(x, -3, 8, 9), lin(y, 5, 7, -8));
        mm[13] = laderman(lin(x, 3, -9), lin(y, 5, -8));
        mm[14] = laderman(x[3], y[7]);
        mm[15] = laderman(lin(x, 8, 9), lin(y, -7, 8));
        mm[16] = laderman(lin(x, -3, 5, 6), lin(y, 6, 7, -9));
        mm[17] = laderman(lin(x, 3, -6), lin(y, 6, -9));
        mm[18] = laderman(lin(x, 5, 6), lin(y, -7, 9));
        mm[19] = laderman(x[2], y[4]);
        mm[20] = laderman(x[6], y[8]);
        mm[21] = laderman(x[4], y[3]);
        mm[22] = laderman(x[7], y[2]);
        mm[23] = laderman(x[9], y[9]);

        double[][][] c = new double[10][][];
        c[1] = lin(mm, 6, 14, 19);
        c[2] = lin(mm, 1, 4, 5, 6, 12, 14, 15);
        c[3] = lin(mm, 6, 7, 9, 10, 14, 16, 18);
        c[4] = lin(mm, 2, 3, 4, 6, 14, 16, 17);
        c[5] = lin(mm, 2, 4, 5, 6, 20);
        c[6] = lin(mm, 14, 16, 17, 18, 21);
        c[7] = lin(mm, 6, 7, 8, 11, 12, 13, 14);
        c[8] = lin(mm, 12, 13, 14, 15, 22);
        c[9] = lin(mm, 6, 7, 8, 9, 23);

        return join(c, 3);
    }

    private static void print(String name, double[][] x) {
        System.out.println(name + " (" + x.length + " x "
                + x[0].length + "):");

        for (double[] row : x) {
            for (double v : row)
                System.out.printf("%8.0f", v);
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Random r = new Random(1);
        double[][] a = new double[5][4];
        double[][] b = new double[4][6];

        for (double[] row : a)
            for (int j = 0; j < row.length; j++)
                row[j] = r.nextInt(10);

        for (double[] row : b)
            for (int j = 0; j < row.length; j++)
                row[j] = r.nextInt(10);

        flops = 0;
        print("Classical", classical(a, b));
        System.out.println("FLOP: " + flops);

        flops = 0;
        print("Binet", binet(a, b));
        System.out.println("FLOP: " + flops);

        flops = 0;
        print("Strassen", strassen(a, b));
        System.out.println("FLOP: " + flops);

        flops = 0;
        print("Laderman", laderman(a, b));
        System.out.println("FLOP: " + flops);
    }
}
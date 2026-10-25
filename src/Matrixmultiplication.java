import java.util.Locale;
import java.util.Random;

/** Java 17+. Macierze kwadratowe n x n, n >= 1; argumenty nie sa modyfikowane.
 * Baza rekurencji: n < 4. W Binecie: wszystkie trzy wymiary bloku < 4.
 * Laderman: J. D. Laderman, Bull. AMS 82 (1976), 126-128.
 */
public final class Matrixmultiplication {
    private static final int CUTOFF = 4;

    private Matrixmultiplication() {
    }

    public static double[][] randomMatrix(int n, Random random) {
        if (n < 1) throw new IllegalArgumentException("n musi byc dodatnie");
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double x;
                // Kontrola obu koncow takze po zaokragleniu double.
                do {
                    x = 1e-8 + (1.0 - 1e-8) * random.nextDouble();
                }
                while (!(x > 1e-8 && x < 1.0));
                a[i][j] = x;
            }
        return a;
    }

    private static void validate(double[][] a, double[][] b) {
        if (a == null || b == null || a.length == 0 || a.length != b.length)
            throw new IllegalArgumentException("Wymagane dwie macierze n x n, n >= 1");
        for (double[][] matrix : new double[][][]{a, b})
            for (double[] row : matrix) {
                if (row == null || row.length != a.length)
                    throw new IllegalArgumentException("Macierz nie jest kwadratowa");
                for (double x : row)
                    if (!Double.isFinite(x))
                        throw new IllegalArgumentException("Elementy musza byc skonczone");
            }
    }

    public static double[][] classical(double[][] a, double[][] b) {
        validate(a, b);
        return direct(a, b);
    }

    private static double[][] direct(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++) c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    public static double[][] binet(double[][] a, double[][] b) {
        validate(a, b);
        int n = a.length;
        double[][] c = new double[n][n];
        binetAdd(a, 0, 0, b, 0, 0, c, 0, 0, n, n, n);
        return c;
    }

    // C += A*B; bloki sa widokami opisanymi indeksami, bez kopiowania.
    // A ma m x k, B ma k x p, C ma m x p.
    private static void binetAdd(double[][] a, int ar, int ac,
                                 double[][] b, int br, int bc,
                                 double[][] c, int cr, int cc,
                                 int m, int k, int p) {
        if (m < CUTOFF && k < CUTOFF && p < CUTOFF) {
            for (int i = 0; i < m; i++)
                for (int t = 0; t < k; t++)
                    for (int j = 0; j < p; j++)
                        c[cr + i][cc + j] += a[ar + i][ac + t] * b[br + t][bc + j];
            return;
        }
        int[] rows = cuts(m), inner = cuts(k), cols = cuts(p);
        // Zwykle 2*2*2 = 8 rekurencyjnych mnozen blokowych.
        for (int i = 0; i < rows.length - 1; i++)
            for (int j = 0; j < cols.length - 1; j++)
                for (int t = 0; t < inner.length - 1; t++)
                    binetAdd(a, ar + rows[i], ac + inner[t],
                            b, br + inner[t], bc + cols[j],
                            c, cr + rows[i], cc + cols[j],
                            rows[i + 1] - rows[i], inner[t + 1] - inner[t],
                            cols[j + 1] - cols[j]);
    }

    private static int[] cuts(int n) {
        return n == 1 ? new int[]{0, 1} : new int[]{0, n / 2, n};
    }

    public static double[][] strassen(double[][] a, double[][] b) {
        validate(a, b);
        return strassenRec(a, b);
    }

    private static double[][] strassenRec(double[][] a, double[][] b) {
        int n = a.length;
        if (n < CUTOFF) return direct(a, b);
        if (n % 2 != 0) {
            // Oddzielenie ostatniego wiersza/kolumny: rdzen (n-1) x (n-1).
            int q = n - 1;
            double[][] core = strassenRec(block(a, 0, 0, q), block(b, 0, 0, q));
            double[][] c = new double[n][n];
            for (int i = 0; i < q; i++)
                for (int j = 0; j < q; j++)
                    c[i][j] = core[i][j] + a[i][q] * b[q][j];
            for (int i = 0; i < q; i++)
                for (int k = 0; k < n; k++)
                    c[i][q] += a[i][k] * b[k][q];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    c[q][j] += a[q][k] * b[k][j];
            return c;
        }

        int s = n / 2;
        double[][][] x = split(a, 2), y = split(b, 2);
        double[][] a11 = x[0], a12 = x[1], a21 = x[2], a22 = x[3];
        double[][] b11 = y[0], b12 = y[1], b21 = y[2], b22 = y[3];
        double[][] m1 = strassenRec(add(a11, a22), add(b11, b22));
        double[][] m2 = strassenRec(add(a21, a22), b11);
        double[][] m3 = strassenRec(a11, sub(b12, b22));
        double[][] m4 = strassenRec(a22, sub(b21, b11));
        double[][] m5 = strassenRec(add(a11, a12), b22);
        double[][] m6 = strassenRec(sub(a21, a11), add(b11, b12));
        double[][] m7 = strassenRec(sub(a12, a22), add(b21, b22));
        double[][] c = new double[n][n];
        put(c, 0, 0, add(sub(add(m1, m4), m5), m7));
        put(c, 0, s, add(m3, m5));
        put(c, s, 0, add(m2, m4));
        put(c, s, s, add(add(sub(m1, m2), m3), m6));
        return c;
    }

    public static double[][] laderman(double[][] a, double[][] b) {
        validate(a, b);
        return ladermanRec(a, b);
    }

    private static double[][] ladermanRec(double[][] a, double[][] b) {
        int n = a.length;
        if (n < CUTOFF) return direct(a, b);
        // Dopelnienie tylko do najblizszej wielokrotnosci 3, na danym poziomie.
        if (n % 3 != 0) {
            int padded = n + (3 - n % 3);
            return block(ladermanRec(pad(a, padded), pad(b, padded)), 0, 0, n);
        }

        int s = n / 3;
        // Numeracja blokow 1..9 odpowiada: 11,12,13,21,22,23,31,32,33.
        double[][][] x = new double[10][][], y = new double[10][][];
        System.arraycopy(split(a, 3), 0, x, 1, 9);
        System.arraycopy(split(b, 3), 0, y, 1, 9);
        double[][][] m = new double[24][][];
        // lin(x,1,-4) oznacza A11-A21. Znak indeksu to znak skladnika.
        m[1] = ladermanRec(lin(x, 1, 2, 3, -4, -5, -8, -9), y[5]);
        m[2] = ladermanRec(lin(x, 1, -4), lin(y, -2, 5));
        m[3] = ladermanRec(x[5], lin(y, -1, 2, 4, -5, -6, -7, 9));
        m[4] = ladermanRec(lin(x, -1, 4, 5), lin(y, 1, -2, 5));
        m[5] = ladermanRec(lin(x, 4, 5), lin(y, -1, 2));
        m[6] = ladermanRec(x[1], y[1]);
        m[7] = ladermanRec(lin(x, -1, 7, 8), lin(y, 1, -3, 6));
        m[8] = ladermanRec(lin(x, -1, 7), lin(y, 3, -6));
        m[9] = ladermanRec(lin(x, 7, 8), lin(y, -1, 3));
        m[10] = ladermanRec(lin(x, 1, 2, 3, -5, -6, -7, -8), y[6]);
        m[11] = ladermanRec(x[8], lin(y, -1, 3, 4, -5, -6, -7, 8));
        m[12] = ladermanRec(lin(x, -3, 8, 9), lin(y, 5, 7, -8));
        m[13] = ladermanRec(lin(x, 3, -9), lin(y, 5, -8));
        m[14] = ladermanRec(x[3], y[7]);
        m[15] = ladermanRec(lin(x, 8, 9), lin(y, -7, 8));
        m[16] = ladermanRec(lin(x, -3, 5, 6), lin(y, 6, 7, -9));
        m[17] = ladermanRec(lin(x, 3, -6), lin(y, 6, -9));
        m[18] = ladermanRec(lin(x, 5, 6), lin(y, -7, 9));
        m[19] = ladermanRec(x[2], y[4]);
        m[20] = ladermanRec(x[6], y[8]);
        m[21] = ladermanRec(x[4], y[3]);
        m[22] = ladermanRec(x[7], y[2]);
        m[23] = ladermanRec(x[9], y[9]);
        double[][] c = new double[n][n];
        put(c, 0, 0, lin(m, 6, 14, 19));
        put(c, 0, s, lin(m, 1, 4, 5, 6, 12, 14, 15));
        put(c, 0, 2 * s, lin(m, 6, 7, 9, 10, 14, 16, 18));
        put(c, s, 0, lin(m, 2, 3, 4, 6, 14, 16, 17));
        put(c, s, s, lin(m, 2, 4, 5, 6, 20));
        put(c, s, 2 * s, lin(m, 14, 16, 17, 18, 21));
        put(c, 2 * s, 0, lin(m, 6, 7, 8, 11, 12, 13, 14));
        put(c, 2 * s, s, lin(m, 12, 13, 14, 15, 22));
        put(c, 2 * s, 2 * s, lin(m, 6, 7, 8, 9, 23));
        return c;
    }

    private static double[][] lin(double[][][] blocks, int... terms) {
        int n = blocks[Math.abs(terms[0])].length;
        double[][] c = new double[n][n];
        for (int term : terms) {
            double[][] b = blocks[Math.abs(term)];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (term > 0) c[i][j] += b[i][j];
                    else c[i][j] -= b[i][j];
        }
        return c;
    }

    private static double[][] add(double[][] a, double[][] b) {
        return lin(new double[][][]{null, a, b}, 1, 2);
    }

    private static double[][] sub(double[][] a, double[][] b) {
        return lin(new double[][][]{null, a, b}, 1, -2);
    }

    private static double[][] block(double[][] a, int row, int col, int n) {
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[row + i], col, b[i], 0, n);
        return b;
    }

    private static double[][][] split(double[][] a, int parts) {
        int s = a.length / parts;
        double[][][] blocks = new double[parts * parts][][];
        for (int i = 0; i < parts; i++)
            for (int j = 0; j < parts; j++)
                blocks[i * parts + j] = block(a, i * s, j * s, s);
        return blocks;
    }

    private static void put(double[][] target, int row, int col, double[][] b) {
        for (int i = 0; i < b.length; i++)
            System.arraycopy(b[i], 0, target[row + i], col, b.length);
    }

    private static double[][] pad(double[][] a, int n) {
        double[][] b = new double[n][n];
        put(b, 0, 0, a);
        return b;
    }

    private static double error(double[][] expected, double[][] actual) {
        double max = 0;
        for (int i = 0; i < expected.length; i++)
            for (int j = 0; j < expected.length; j++) {
                double d = Math.abs(expected[i][j] - actual[i][j]);
                if (!Double.isFinite(d)) throw new AssertionError("Nieskonczony wynik");
                max = Math.max(max, d);
                if (d > 1e-10 + 1e-10 * Math.abs(expected[i][j]))
                    throw new AssertionError("Roznica w [" + i + "][" + j + "]: " + d);
            }
        return max;
    }

    private static void checkAll(double[][] a, double[][] b) {
        double[][] expected = classical(a, b);
        error(expected, binet(a, b));
        error(expected, strassen(a, b));
        error(expected, laderman(a, b));
    }

    public static void selfTest() {
        Random r = new Random(2026);
        int cases = 0;
        for (int n = 1; n <= 40; n++) {
            checkAll(randomMatrix(n, r), randomMatrix(n, r));
            cases++;
            double[][] a = new double[n][n], b = new double[n][n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    a[i][j] = r.nextInt(11) - 5;
                    b[i][j] = r.nextInt(11) - 5;
                }
            checkAll(a, b);
            cases++;
        }
        for (int n : new int[]{63, 64, 65, 81, 100}) {
            checkAll(randomMatrix(n, r), randomMatrix(n, r));
            cases++;
        }

        // 729 par blokow bazowych: kontrola wszystkich wspolczynnikow Ladermana.
        // n=6 uruchamia 23 mnozenia blokow 2x2 (n=3 konczy sie w bazie).
        for (int u = 0; u < 9; u++)
            for (int v = 0; v < 9; v++)
                for (int t = 0; t < 9; t++) {
                    double[][] a = new double[6][6], b = new double[6][6];
                    a[2 * (u / 3)][2 * (u % 3)] = 1;
                    b[2 * (v / 3)][2 * (v % 3)] = 1;
                    // Kazdy z 9 blokow wyjsciowych sprawdzany osobno.
                    double[][] c = laderman(a, b), e = classical(a, b);
                    int i = 2 * (t / 3), j = 2 * (t % 3);
                    if (c[i][j] != e[i][j]) throw new AssertionError("Wzory Ladermana");
                }
        double[][] a = randomMatrix(7, r), original = block(a, 0, 0, 7);
        checkAll(a, a);
        error(original, a);
        System.out.println("OK: " + cases + " par macierzy, 729 kontroli wspolczynnikow Ladermana, A*A i brak modyfikacji A.");
    }

    private static void print(double[][] a) {
        for (double[] row : a) {
            for (double x : row) System.out.printf(Locale.ROOT, "%11.7f ", x);
            System.out.println();
        }
    }

    // ===== POMIARY DO RAPORTU (algorytmy powyzej bez zmian) =====
    private static volatile double[][] benchmarkSink;

// Operacje arytmetyczne double w kodzie zrodlowym, nie instrukcje CPU.
// += iloczyn to 2 FLOP; liczymy tez dodawanie do poczatkowego zera.

    private static long flopCount(int method, int n) {
        if (method == 0 || n < CUTOFF) return 2L * n * n * n;
        if (method == 1) {
            if (n % 2 == 0) {
                long s = n / 2;
                // 18 wywolan add/sub; kazde lin wykonuje 2 operacje na element.
                return 7 * flopCount(1, n / 2) + 36 * s * s;
            }
            long q = n - 1;
            return flopCount(1, n - 1) + 2 * (q * q + q * n + (long) n * n);
        }
        if (n % 3 != 0) return flopCount(2, n + 3 - n % 3);
        long s = n / 3;
        // Wszystkie lin w ladermanRec: lacznie 135 skladnikow na element bloku.
        return 23 * flopCount(2, n / 3) + 135 * s * s;
    }

    private static double[][] runMethod(int method, double[][] a, double[][] b) {
        // Bez validate: walidacja wejsc nie jest czescia pomiaru algorytmu.
        if (method == 1) return strassenRec(a, b);
        if (method == 2) return ladermanRec(a, b);
        int n = a.length;
        double[][] c = new double[n][n];
        binetAdd(a, 0, 0, b, 0, 0, c, 0, 0, n, n, n);
        return c;
    }

    private static void writeEnvironment(java.nio.file.Path path, int maxN, int repeats,
                                         double budget, long seed) throws java.io.IOException {
        String cpu = System.getenv("PROCESSOR_IDENTIFIER");
        java.nio.file.Path proc = java.nio.file.Path.of("/proc/cpuinfo");
        if (java.nio.file.Files.isReadable(proc)) {
            for (String line : java.nio.file.Files.readAllLines(proc))
                if (line.startsWith("model name")) {
                    cpu = line.substring(line.indexOf(':') + 1).trim();
                    break;
                }
        }
        if (cpu == null) cpu = "Sprawdz model CPU w ustawieniach systemu";
        java.lang.management.OperatingSystemMXBean os =
                java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        long ram = os instanceof com.sun.management.OperatingSystemMXBean
                ? ((com.sun.management.OperatingSystemMXBean) os).getTotalMemorySize() : -1;
        String text = "Data: " + java.time.ZonedDateTime.now() + "\n"
                + "CPU (identyfikator; zweryfikuj pelny model w systemie): " + cpu + "\n"
                + "Procesory logiczne dostepne JVM: " + Runtime.getRuntime().availableProcessors() + "\n"
                + "Pamiec raportowana przez OS/JVM [B]: " + ram + "\n"
                + "Limit sterty JVM [B], NIE rozmiar RAM: " + Runtime.getRuntime().maxMemory() + "\n"
                + "OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " " + System.getProperty("os.arch") + "\n"
                + "Java: " + System.getProperty("java.runtime.version") + "\n"
                + "JVM: " + System.getProperty("java.vm.name") + " " + System.getProperty("java.vm.info") + "\n"
                + "Argumenty JVM: " + java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments() + "\n"
                + "maxN=" + maxN + ", repeats=" + repeats + ", budgetSeconds=" + budget + ", seed=" + seed + "\n"
                + "CUTOFF=" + CUTOFF + "; rozgrzewka >= 2 s; mediana powtorzen.\n"
                + "Dla n<=16 probka to srednia ze 100 mnozen, dla pozostalych jedno mnozenie.\n"
                + "Czas obejmuje alokacje, kopiowanie, ewentualny GC i zapis wyniku do volatile.\n"
                + "Poza czasem: losowanie, walidacja, kontrola wyniku, CSV, SVG, liczenie FLOP.\n"
                + "FLOP: liczba +, -, * na double w kodzie; nie sprzetowy licznik instrukcji.\n"
                + "Dodawanie do zera tez liczone; kopiowanie, indeksy, porownania pomijane.\n"
                + "Limit czasu sprawdzany miedzy rozmiarami, nie przerywa trwajacego mnozenia.\n";
        java.nio.file.Files.writeString(path, text, java.nio.charset.StandardCharsets.UTF_8);
    }
// Samodzielne wykresy SVG, bez dodatkowych bibliotek i bez Pythona.

    private static void writePlot(java.nio.file.Path path, java.util.List<double[]> rows,
                                  boolean flops) throws java.io.IOException {
        if (rows.isEmpty()) return;
        int offset = flops ? 4 : 1;
        double ymax = 0;
        for (double[] row : rows) for (int k = 0; k < 3; k++) ymax = Math.max(ymax, row[offset + k]);
        ymax = ymax == 0 ? 1 : ymax * 1.05;
        double xmax = Math.max(2, rows.get(rows.size() - 1)[0]);
        String[] colors = {"#2563eb", "#dc2626", "#059669"};
        String[] names = {"Binet", "Strassen", "Laderman"};
        StringBuilder s = new StringBuilder("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"620\" viewBox=\"0 0 1000 620\"><rect width=\"1000\" height=\"620\" fill=\"white\"/><g font-family=\"sans-serif\" font-size=\"15\" fill=\"#111827\">");
        s.append("<text x=\"90\" y=\"30\" font-size=\"22\">")
                .append(flops ? "Liczba operacji zmiennoprzecinkowych" : "Czas mnozenia — mediana pomiarow")
                .append("</text>");
        for (int t = 0; t <= 5; t++) {
            double y = 520 - 440.0 * t / 5, val = ymax * t / 5;
            s.append(String.format(Locale.ROOT, "<path d=\"M 100 %.2f H 960\" stroke=\"#e5e7eb\"/><text x=\"90\" y=\"%.2f\" text-anchor=\"end\">%.3g</text>", y, y + 5, val));
            double x = 100 + 860.0 * t / 5, n = 1 + (xmax - 1) * t / 5;
            s.append(String.format(Locale.ROOT, "<text x=\"%.2f\" y=\"545\" text-anchor=\"middle\">%.0f</text>", x, n));
        }
        s.append("<path d=\"M 100 80 V 520 H 960\" stroke=\"#374151\" fill=\"none\"/>")
                .append("<text x=\"450\" y=\"580\">Rozmiar macierzy n (n x n)</text>")
                .append("<text x=\"100\" y=\"65\">").append(flops ? "FLOP" : "Czas [ms]").append("</text>");
        for (int k = 0; k < 3; k++) {
            s.append("<polyline fill=\"none\" stroke=\"").append(colors[k]).append("\" stroke-width=\"1.6\" points=\"");
            for (double[] row : rows)
                s.append(String.format(Locale.ROOT, "%.2f,%.2f ", 100 + 860 * (row[0] - 1) / (xmax - 1), 520 - 440 * row[offset + k] / ymax));
            s.append("\"/><text x=\"").append(520 + 145 * k).append("\" y=\"62\" fill=\"")
                    .append(colors[k]).append("\">").append(names[k]).append("</text>");
        }

        s.append("</g></svg>");
        java.nio.file.Files.writeString(path, s, java.nio.charset.StandardCharsets.UTF_8);
    }

    public static void main(String[] args) throws java.io.IOException {
        if (args.length > 0 && args[0].equals("--test")) {
            selfTest();
            return;
        }
        int maxN = args.length > 0 ? Integer.parseInt(args[0]) : 200;
        int repeats = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        double budget = args.length > 2 ? Double.parseDouble(args[2]) : 600;
        java.nio.file.Path out = java.nio.file.Path.of(args.length > 3 ? args[3] : "wyniki");
        if (maxN < 1 || maxN > 1000 || repeats < 1 || !Double.isFinite(budget) || budget <= 0)
            throw new IllegalArgumentException("Argumenty: maxN [1..1000], repeats >=1, budzetSekund >0, katalog");
        java.nio.file.Files.createDirectories(out);
        // Nowy katalog dla kazdego uruchomienia, bez nadpisywania poprzednich wynikow.
        out = java.nio.file.Files.createTempDirectory(out, "pomiar-");
        long seed = 42;
        writeEnvironment(out.resolve("srodowisko.txt"), maxN, repeats, budget, seed);
        System.out.println("Wyniki: " + out.toAbsolutePath());
        System.out.println("Rozgrzewka JVM...");
        Random warm = new Random(2026);
        long warmEnd = System.nanoTime() + 2_000_000_000L;
        int iteration = 0;
        do {
            int n = new int[]{2, 3, 32, 33, 36}[iteration++ % 5];
            double[][] a = randomMatrix(n, warm), b = randomMatrix(n, warm);
            for (int method = 0; method < 3; method++) benchmarkSink = runMethod(method, a, b);
        } while (System.nanoTime() < warmEnd);
        java.util.List<double[]> rows = new java.util.ArrayList<>();
        Random random = new Random(seed);
        long start = System.nanoTime();
        try (java.io.BufferedWriter csv = java.nio.file.Files.newBufferedWriter(out.resolve("pomiary.csv"));
             java.io.BufferedWriter raw = java.nio.file.Files.newBufferedWriter(out.resolve("proby.csv"))) {
            csv.write("n,binet_ms,strassen_ms,laderman_ms,binet_flop,strassen_flop,laderman_flop\n");
            raw.write("n,metoda,powtorzenie,mnozen_w_probce,ms_na_mnozenie\n");
            for (int n = 1; n <= maxN; n++) {
                if (n > 1 && (System.nanoTime() - start) / 1e9 >= budget) {
                    System.out.println("Osiagnieto budzet czasu; ostatni rozmiar: " + (n - 1));
                    break;
                }

                double[][] a = randomMatrix(n, random), b = randomMatrix(n, random);
                validate(a, b);
                double[][] times = new double[3][repeats];
                int batch = n <= 16 ? 100 : 1;
                for (int r = 0; r < repeats; r++)
                    for (int order = 0; order < 3; order++) {
                        int method = (n + r + order) % 3; // rotacja kolejnosci metod
                        double[][] c = null;
                        long t = System.nanoTime();
                        for (int k = 0; k < batch; k++) {
                            c = runMethod(method, a, b);
                            benchmarkSink = c; // wynik ucieka; zapis volatile wliczony w czas
                        }
                        times[method][r] = (System.nanoTime() - t) / 1e6 / batch;
                        // Kontrola kilku elementow poza mierzonym odcinkiem, O(n).
                        for (int pos : new int[]{0, n / 2, n - 1}) {
                            double expected = 0;
                            for (int k = 0; k < n; k++) expected += a[pos][k] * b[k][pos];
                            if (!Double.isFinite(c[pos][pos]) || Math.abs(c[pos][pos] - expected) > 1e-9 * (1 + Math.abs(expected)))
                                throw new AssertionError("Blad wyniku, n=" + n + ", metoda=" + method);
                        }
                        raw.write(String.format(Locale.ROOT, "%d,%s,%d,%d,%.9f%n", n,
                                new String[]{"Binet", "Strassen", "Laderman"}[method], r + 1, batch, times[method][r]));
                    }
                double[] row = new double[7];
                row[0] = n;
                for (int method = 0; method < 3; method++) {
                    java.util.Arrays.sort(times[method]);
                    row[1 + method] = (times[method][(repeats - 1) / 2] + times[method][repeats / 2]) / 2;
                    row[4 + method] = flopCount(method, n);
                }
                csv.write(String.format(Locale.ROOT, "%d,%.9f,%.9f,%.9f,%d,%d,%d%n", n,
                        row[1], row[2], row[3], flopCount(0, n), flopCount(1, n), flopCount(2, n)));
                csv.flush();
                raw.flush();
                rows.add(row);
                // Zapis po kazdym n: kompletne dotychczasowe wykresy nawet po przerwaniu.
                writePlot(out.resolve("czas.svg"), rows, false);
                writePlot(out.resolve("operacje.svg"), rows, true);
                if (n % 10 == 0 || n == maxN) System.out.printf(Locale.ROOT,
                        "n=%d: Binet %.3f ms, Strassen %.3f ms, Laderman %.3f ms%n", n, row[1], row[2], row[3]);
            }
        }
        System.out.println("Gotowe: pomiary.csv, proby.csv, czas.svg, operacje.svg, srodowisko.txt");
    }
}

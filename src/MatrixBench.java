import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/** Pomiary do sprawozdania: czas i liczba operacji zmiennoprzecinkowych dla n = 1..maxN.
 * Uzycie:  java MatrixBench [maxN=1000] [powtorzenia=3] [budzet_s=3600] [cutoff=64]
 * Dla kazdego n czas to mediana z "powtorzenia" probek. Pelny zakres do 1000 trwa kilkanascie
 * do kilkudziesieciu minut; szybciej: java MatrixBench 1000 1. Cutoff 2 to "czysta" rekurencja,
 * ale dla duzych n jest nieobliczalna w sensownym czasie.
 * Wyniki trafiaja do katalogu dane/ (pomiary.csv, srodowisko.txt).
 * Wymaga MatrixMultAlt.java w tym samym katalogu.
 */
public final class MatrixBench {
    private static final String[] NAMES = {"classical", "binet", "strassen", "laderman"};
    private static final Path OUT = Path.of("dane");
    private static volatile double[][] sink; // wynik "ucieka", wiec JIT nie wytnie obliczen

    private static double[][] run(int method, double[][] a, double[][] b) {
        return switch (method) {
            case 0 -> MatrixMultAlt.classical(a, b);
            case 1 -> MatrixMultAlt.binet(a, b);
            case 2 -> MatrixMultAlt.strassen(a, b);
            default -> MatrixMultAlt.laderman(a, b);
        };
    }

    private static double[][] random(int n, Random r) {
        double[][] a = new double[n][n];
        for (double[] row : a) for (int j = 0; j < n; j++) row[j] = r.nextDouble();
        return a;
    }

    public static void main(String[] args) throws Exception {
        int maxN = args.length > 0 ? Integer.parseInt(args[0]) : 1000;
        int repeats = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        double budget = args.length > 2 ? Double.parseDouble(args[2]) : 3600;
        MatrixMultAlt.CUTOFF = args.length > 3 ? Integer.parseInt(args[3]) : 64;
        Files.createDirectories(OUT);
        writeEnvironment(maxN, repeats, budget);
        System.out.println("Rozgrzewka JVM (3 s)...");
        warmUp();
        measure(maxN, repeats, budget);
    }

    // ===== Rozgrzewka: JIT ma skompilowac petle, zanim zaczniemy mierzyc =====

    private static void warmUp() {
        Random r = new Random(1);
        int[] sizes = {5, 17, 33, 64, 100, 130, 200};
        long end = System.nanoTime() + 3_000_000_000L;
        int it = 0;
        do {
            int n = sizes[it++ % sizes.length];
            double[][] a = random(n, r), b = random(n, r);
            for (int m = 0; m < 4; m++) sink = run(m, a, b);
        } while (System.nanoTime() < end);
    }

    // ===== Pomiary =====

    private static void measure(int maxN, int repeats, double budget) throws IOException {
        Random rnd = new Random(42);
        long start = System.nanoTime();
        int lastN = 0;
        try (var w = Files.newBufferedWriter(OUT.resolve("pomiary.csv"))) {
            w.write("n,classical_ms,binet_ms,strassen_ms,laderman_ms,"
                    + "classical_flop,binet_flop,strassen_flop,laderman_flop\n");
            for (int n = 1; n <= maxN; n++) {
                if (n > 1 && (System.nanoTime() - start) / 1e9 >= budget) {
                    System.out.println("Wyczerpany budzet czasu, ostatni rozmiar: " + (n - 1));
                    break;
                }
                double[][] a = random(n, rnd), b = random(n, rnd);
                // Dla malych n jedna probka to srednia z wielu mnozen (zegar ma skonczona rozdzielczosc).
                int batch = (int) Math.max(1, 1_000_000L / ((long) n * n * n));
                double[][] ms = new double[4][repeats];
                long[] flop = new long[4];
                double[][][] last = new double[4][][];
                for (int r = 0; r < repeats; r++)
                    for (int o = 0; o < 4; o++) {
                        int m = (n + r + o) % 4; // rotacja kolejnosci metod
                        MatrixMultAlt.flops = 0;
                        double[][] c = null;
                        long t0 = System.nanoTime();
                        for (int k = 0; k < batch; k++) {
                            c = run(m, a, b);
                            sink = c;
                        }
                        ms[m][r] = (System.nanoTime() - t0) / 1e6 / batch;
                        flop[m] = MatrixMultAlt.flops / batch;
                        last[m] = c;
                    }
                check(n, last, flop);
                StringBuilder row = new StringBuilder().append(n);
                for (int m = 0; m < 4; m++) row.append(',').append(median(ms[m]));
                for (int m = 0; m < 4; m++) row.append(',').append(flop[m]);
                w.write(row.append('\n').toString());
                w.flush(); // po przerwaniu zostaja dane dotychczasowe
                lastN = n;
                if (n % 20 == 0) System.out.printf("n=%d  binet %.3f ms  strassen %.3f ms  laderman %.3f ms%n",
                        n, median(ms[1]), median(ms[2]), median(ms[3]));
            }
        }
        double sec = (System.nanoTime() - start) / 1e9;
        Files.writeString(OUT.resolve("srodowisko.txt"),
                "Zakres pomiarow: n = 1.." + lastN + "\n"
                        + "Czas pomiarow [s]: " + Math.round(sec) + "\n"
                        + "Laczny czas kompilacji JIT [ms]: "
                        + ManagementFactory.getCompilationMXBean().getTotalCompilationTime() + "\n",
                java.nio.file.StandardOpenOption.APPEND);
        System.out.println("Gotowe: dane/pomiary.csv (n = 1.." + lastN + ")");
    }

    private static double median(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        return (s[(s.length - 1) / 2] + s[s.length / 2]) / 2;
    }

    /** Kontrola poprawnosci: wyniki zgodne z klasycznym, a licznik klasycznego rowny 2n^3. */
    private static void check(int n, double[][][] c, long[] flop) {
        if (flop[0] != 2L * n * n * n) throw new AssertionError("Licznik FLOP klasycznego, n=" + n);
        for (int m = 1; m < 4; m++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (Math.abs(c[m][i][j] - c[0][i][j]) > 1e-8 * (1 + Math.abs(c[0][i][j])))
                        throw new AssertionError("Zly wynik: " + NAMES[m] + ", n=" + n);
    }

    // ===== Opis srodowiska =====

    private static void writeEnvironment(int maxN, int repeats, double budget) throws IOException {
        var os = ManagementFactory.getOperatingSystemMXBean();
        long ram = os instanceof com.sun.management.OperatingSystemMXBean x ? x.getTotalMemorySize() : -1;
        String text = "Procesor: " + cpuName() + "\n"
                + "Rdzenie logiczne: " + Runtime.getRuntime().availableProcessors() + "\n"
                + "RAM [MB]: " + (ram > 0 ? ram / (1024 * 1024) : "nieznany") + "\n"
                + "Limit sterty JVM [MB]: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + "\n"
                + "System: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " " + System.getProperty("os.arch") + "\n"
                + "Java: " + System.getProperty("java.runtime.version") + "\n"
                + "JVM: " + System.getProperty("java.vm.name") + " (" + System.getProperty("java.vm.info") + ")\n"
                + "Kompilator JIT: " + ManagementFactory.getCompilationMXBean().getName() + "\n"
                + "Parametry: maxN=" + maxN + ", powtorzenia=" + repeats + ", budzet=" + budget
                + " s, CUTOFF=" + MatrixMultAlt.CUTOFF + "\n";
        Files.writeString(OUT.resolve("srodowisko.txt"), text);
    }

    private static String cpuName() {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            if (os.contains("win"))
                return firstLine("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name");
            if (os.contains("mac"))
                return firstLine("sysctl", "-n", "machdep.cpu.brand_string");
            for (String line : Files.readAllLines(Path.of("/proc/cpuinfo")))
                if (line.startsWith("model name")) return line.substring(line.indexOf(':') + 1).trim();
        } catch (Exception ignored) {
            // przechodzimy do zmiennej srodowiskowej
        }
        String id = System.getenv("PROCESSOR_IDENTIFIER");
        return id != null ? id : "nieznany";
    }

    private static String firstLine(String... cmd) throws Exception {
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes()).trim();
        p.waitFor();
        return out.lines().findFirst().orElse("nieznany").trim();
    }
}

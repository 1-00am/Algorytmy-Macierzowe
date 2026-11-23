import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Pomiary rekurencyjnego odwracania i LU zgodnych ze slajdami.
 * Java 17+. Wymaga MatrixMultAlt.java i poprawionego MatrixRecursive.java.
 * Uzycie: java MatrixRecursiveBench [maxN=1000] [powtorzenia=3]
 *          [budzet_s=3600] [cutoff=2] [katalog=dane_recursive]
 * Czas: mediana srednich czasow pojedynczej operacji w paczkach.
 * Budzet obejmuje pomiary i kontrole, ale nie rozgrzewke JVM.
 * Jest sprawdzany miedzy rozmiarami; biezacy rozmiar jest dokonczony.
 */
public final class MatrixRecursiveBench {
    private static final String[] NAMES = {
            "inverse_binet", "inverse_strassen", "inverse_laderman",
            "lu_binet", "lu_strassen"
    };
    private static final int COUNT = NAMES.length;
    private static final int PROBES = 4;
    private static final double TOLERANCE = 1e-8;
    private static volatile Object sink;

    private MatrixRecursiveBench() {
    }

    private static Object run(int method, double[][] a) {
        return switch (method) {
            case 0 -> MatrixRecursive.inverse(a, MatrixRecursive.Method.BINET);
            case 1 -> MatrixRecursive.inverse(a, MatrixRecursive.Method.STRASSEN);
            case 2 -> MatrixRecursive.inverse(a, MatrixRecursive.Method.LADERMAN);
            case 3 -> MatrixRecursive.lu(a, MatrixRecursive.Method.BINET);
            case 4 -> MatrixRecursive.lu(a, MatrixRecursive.Method.STRASSEN);
            default -> throw new IllegalArgumentException("Nieznana metoda.");
        };
    }

    private static MatrixRecursive.Stats stats(Object result) {
        if (result instanceof MatrixRecursive.InverseResult x) return x.stats;
        return ((MatrixRecursive.LUResult) result).stats;
    }

    // Dominacja przekatniowa zapewnia nieosobliwosc blokow wymaganych
    // przez wariant bez pivotingu. Elementy sa dodatnie i mniejsze od 1.
    private static double[][] randomMatrix(int n, Random random) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = i == j
                        ? 0.5 + 0.5 * random.nextDouble()
                        : (0.05 + 0.15 * random.nextDouble()) / n;
        return a;
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 5)
            throw new IllegalArgumentException("Uzycie: java MatrixRecursiveBench "
                    + "[maxN] [powtorzenia] [budzet_s] [cutoff] [katalog]");
        int maxN = args.length > 0 ? Integer.parseInt(args[0]) : 1000;
        int repeats = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        double budget = args.length > 2 ? Double.parseDouble(args[2]) : 3600;
        int cutoff = args.length > 3 ? Integer.parseInt(args[3]) : 2;
        if (maxN < 1 || repeats < 1 || cutoff < 1
                || !Double.isFinite(budget) || budget <= 0)
            throw new IllegalArgumentException("Wszystkie parametry musza byc dodatnie.");

        MatrixMultAlt.CUTOFF = cutoff;
        Path base = Path.of(args.length > 4 ? args[4] : "dane_recursive");
        Files.createDirectories(base);
        String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
        Path out = Files.createTempDirectory(base, stamp + "-");
        writeEnvironment(out, maxN, repeats, budget);

        System.out.println("Wyniki: " + out.toAbsolutePath());
        System.out.println("Rozgrzewka JVM (okolo 3 s)...");
        warmUp(maxN);
        measure(out, maxN, repeats, budget);
    }

    private static void warmUp(int maxN) {
        Random random = new Random(1);
        int[] sizes = {5, 17, 33, 65, 132};
        long start = System.nanoTime();
        int iteration = 0;
        do {
            int n = Math.min(maxN, sizes[iteration++ % sizes.length]);
            double[][] a = randomMatrix(n, random);
            for (int m = 0; m < COUNT; m++) {
                sink = run(m, a);
                if (secondsSince(start) >= 3.0) return;
            }
        } while (secondsSince(start) < 3.0);
    }

    private static void measure(Path out, int maxN, int repeats,
                                double budget) throws IOException {
        Random random = new Random(42);
        long start = System.nanoTime();
        int lastN = 0;
        String status = "pelny zakres";

        try (BufferedWriter writer = Files.newBufferedWriter(out.resolve("pomiary.csv"))) {
            StringBuilder header = new StringBuilder("n,batch");
            for (String suffix : new String[]{"_ms", "_flop_mul", "_flop_other", "_flop_total", "_check_error"})
                for (String name : NAMES) header.append(',').append(name).append(suffix);
            writer.write(header.append('\n').toString());
            writer.flush();

            for (int n = 1; n <= maxN; n++) {
                if (secondsSince(start) >= budget) {
                    status = "wyczerpany budzet";
                    break;
                }

                // Ta sama macierz dla wszystkich metod i powtorzen danego n.
                double[][] a = randomMatrix(n, random);
                int batch = (int) Math.max(1,
                        Math.min(4096, 100_000.0 / ((double) n * n * n)));
                double[][] times = new double[COUNT][repeats];
                long[][] flops = new long[COUNT][3];
                Object[] last = new Object[COUNT];

                for (int r = 0; r < repeats; r++) {
                    for (int order = 0; order < COUNT; order++) {
                        int m = (n + r + order) % COUNT;
                        MatrixMultAlt.flops = 0;
                        Object result = null;

                        long t0 = System.nanoTime();
                        for (int k = 0; k < batch; k++) {
                            result = run(m, a);
                            sink = result;
                        }
                        times[m][r] = (System.nanoTime() - t0) / 1e6 / batch;

                        // Stats dotyczy JEDNEGO wywolania, wiec nie dzielimy go przez batch.
                        MatrixRecursive.Stats s = stats(result);
                        if (MatrixMultAlt.flops != s.multiplicationFlops * batch)
                            throw new AssertionError("Niezgodny licznik mnozen: " + NAMES[m]);
                        if (r > 0 && (flops[m][0] != s.multiplicationFlops
                                || flops[m][1] != s.otherFlops))
                            throw new AssertionError("Niezgodne liczniki miedzy probkami.");
                        flops[m][0] = s.multiplicationFlops;
                        flops[m][1] = s.otherFlops;
                        flops[m][2] = s.totalFlops();
                        last[m] = result;
                    }
                }

                // Kontrola poza pomiarem czasu i poza licznikami FLOP.
                double[] errors = new double[COUNT];
                for (int m = 0; m < COUNT; m++) {
                    errors[m] = check(a, last[m]);
                    if (!Double.isFinite(errors[m]) || errors[m] > TOLERANCE)
                        throw new AssertionError("Zly wynik: " + NAMES[m]
                                + ", n=" + n + ", blad=" + errors[m]);
                }

                StringBuilder row = new StringBuilder().append(n).append(',').append(batch);
                for (double[] samples : times) row.append(',').append(median(samples));
                for (int kind = 0; kind < 3; kind++)
                    for (int m = 0; m < COUNT; m++) row.append(',').append(flops[m][kind]);
                for (double error : errors) row.append(',').append(error);
                writer.write(row.append('\n').toString());
                writer.flush();
                lastN = n;

                if (n == 1 || n % 20 == 0 || n == maxN)
                    System.out.printf(Locale.ROOT,
                            "n=%d | inverse: %.3f / %.3f / %.3f ms | LU: %.3f / %.3f ms%n",
                            n, median(times[0]), median(times[1]), median(times[2]),
                            median(times[3]), median(times[4]));
            }
        } catch (IOException | RuntimeException | Error failure) {
            status = "przerwano: " + failure;
            throw failure;
        } finally {
            var jit = ManagementFactory.getCompilationMXBean();
            String jitTime = jit != null && jit.isCompilationTimeMonitoringSupported()
                    ? Long.toString(jit.getTotalCompilationTime()) : "niedostepny";
            Files.writeString(out.resolve("srodowisko.txt"),
                    "\nStatus: " + status
                            + "\nOstatni zapisany rozmiar: " + lastN
                            + "\nCzas pomiarow i kontroli [s]: " + secondsSince(start)
                            + "\nCzas kompilacji JIT od startu JVM [ms]: " + jitTime + "\n",
                    StandardOpenOption.APPEND);
        }
        System.out.println("Gotowe: " + out.resolve("pomiary.csv") + ", zapisano " + lastN + " rozmiarow.");
    }

    private static double secondsSince(long start) {
        return (System.nanoTime() - start) / 1e9;
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return (sorted[(sorted.length - 1) / 2] + sorted[sorted.length / 2]) / 2;
    }

    // Test na 4 wektorach znakowych: O(n^2), zamiast pelnego mnozenia O(n^3).
    // Jest kontrola probkowana, a nie sprawdzeniem wszystkich elementow iloczynu.
    private static double check(double[][] a, Object result) {
        int n = a.length;
        Random random = new Random(1234567L + n);
        double error = 0;
        for (int probe = 0; probe < PROBES; probe++) {
            double[] v = new double[n];
            for (int i = 0; i < n; i++) v[i] = random.nextBoolean() ? 1.0 : -1.0;
            if (result instanceof MatrixRecursive.InverseResult x) {
                error = Math.max(error, relativeError(mv(a, mv(x.inverse, v)), v));
                error = Math.max(error, relativeError(mv(x.inverse, mv(a, v)), v));
            } else {
                MatrixRecursive.LUResult x = (MatrixRecursive.LUResult) result;
                error = Math.max(error, relativeError(mv(x.L, mv(x.U, v)), mv(a, v)));
            }
        }
        return error;
    }

    private static double[] mv(double[][] a, double[] v) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < v.length; j++) result[i] += a[i][j] * v[j];
        return result;
    }

    private static double relativeError(double[] actual, double[] expected) {
        double error = 0, scale = 1;
        for (int i = 0; i < actual.length; i++) {
            if (!Double.isFinite(actual[i])) return Double.POSITIVE_INFINITY;
            error = Math.max(error, Math.abs(actual[i] - expected[i]));
            scale = Math.max(scale, Math.abs(expected[i]));
        }
        return error / scale;
    }

    private static void writeEnvironment(Path out, int maxN, int repeats,
                                         double budget) throws IOException {
        var os = ManagementFactory.getOperatingSystemMXBean();
        var jit = ManagementFactory.getCompilationMXBean();
        long ram = os instanceof com.sun.management.OperatingSystemMXBean x
                ? x.getTotalMemorySize() : -1;
        String text = "Algorytm: " + MatrixRecursive.IMPLEMENTATION
                + "\nProcesor: " + cpuName()
                + "\nRdzenie logiczne: " + Runtime.getRuntime().availableProcessors()
                + "\nRAM [MB]: " + (ram > 0 ? ram / (1024 * 1024) : "nieznany")
                + "\nLimit sterty JVM [MB]: " + Runtime.getRuntime().maxMemory() / (1024 * 1024)
                + "\nSystem: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " " + System.getProperty("os.arch")
                + "\nJava: " + System.getProperty("java.runtime.version")
                + "\nJVM: " + System.getProperty("java.vm.name") + " (" + System.getProperty("java.vm.info") + ")"
                + "\nKompilator JIT: " + (jit == null ? "brak" : jit.getName())
                + "\nArgumenty JVM: " + ManagementFactory.getRuntimeMXBean().getInputArguments()
                + "\nParametry: maxN=" + maxN + ", powtorzenia=" + repeats
                + ", budzet=" + budget + " s, CUTOFF=" + MatrixMultAlt.CUTOFF
                + "\nDane: seed=42, a_ii=0.5+0.5*U, a_ij=(0.05+0.15*U)/n dla i!=j."
                + "\nU to niezalezne losowanie z [0,1); macierze scisle dominujace przekatniowo."
                + "\nCzas obejmuje cale wywolanie API: kontrole wejscia, alokacje, liczniki i obliczenia."
                + "\nGenerowanie macierzy i kontrola wyniku sa poza pomiarem czasu operacji."
                + "\nKontrola: 4 losowe wektory +/-1, oba kierunki dla inverse; tolerancja=" + TOLERANCE
                + "\ncheck_error = max ||wynik-oczekiwany||_inf / max(1,||oczekiwany||_inf)."
                + "\nFLOP: +,-,*,/; osobno mnozenia, operacje pozostale i suma; bez kontroli wyniku."
                + "\nCUTOFF dotyczy mnozen; inverse i LU dziela bloki az do rozmiaru 1."
                + "\nBudzet jest miekki, sprawdzany miedzy rozmiarami; rozgrzewka okolo 3 s poza budzetem.\n";
        Files.writeString(out.resolve("srodowisko.txt"), text);
    }

    private static String cpuName() {
        try {
            String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
            if (os.contains("win"))
                return firstLine("powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "(Get-CimInstance Win32_Processor).Name");
            if (os.contains("mac"))
                return firstLine("sysctl", "-n", "machdep.cpu.brand_string");
            for (String line : Files.readAllLines(Path.of("/proc/cpuinfo")))
                if (line.startsWith("model name"))
                    return line.substring(line.indexOf(':') + 1).trim();
        } catch (Exception ignored) {
            // Zapasowe zrodlo informacji.
        }
        String name = System.getenv("PROCESSOR_IDENTIFIER");
        return name == null ? "nieznany" : name;
    }

    private static String firstLine(String... command) throws Exception {
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            if (!process.waitFor(3, TimeUnit.SECONDS))
                throw new IOException("Przekroczony czas odczytu modelu CPU.");
            String text = new String(process.getInputStream().readAllBytes()).trim();
            if (process.exitValue() != 0 || text.isEmpty())
                throw new IOException("Nie udalo sie odczytac modelu CPU.");
            return text.lines().findFirst().orElse("nieznany").trim();
        } finally {
            if (process.isAlive()) process.destroyForcibly();
        }
    }
}
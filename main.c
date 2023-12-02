#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>

// Definiere die Funktion für die Riemann p-Energie
double p_energy(double u, double u_prime, double p) {
    return 0.5 * pow(fabs(u_prime), p);
}

// Definiere die Ableitung der Riemann p-Energie
double p_energy_derivative(double u, double u_prime, double p) {
    return p * pow(fabs(u_prime), p - 2) * ((u_prime > 0) ? 1 : -1);
}

// Definiere die Funktion, die der Euler-Lagrange-Gleichung entspricht
double lagrange_equation(double u, double u_prime, double p) {
    return gsl_deriv_central(p_energy, u, 1e-8, p, u_prime);
}

// Hilfsfunktion für die GSL-Minimierung
double lagrange_equation_root(double u, void *params) {
    double p = *(double *)params;
    return lagrange_equation(u, gsl_deriv_central(lagrange_equation, u, 1e-8, p), p);
}

int main() {
    const double p = 2.0;

    gsl_min_fminimizer *minimizer = gsl_min_fminimizer_alloc(gsl_min_fminimizer_brent);
    gsl_function F;
    F.function = &lagrange_equation_root;
    F.params = &p;

    double initial_guess = 1.0;
    gsl_min_fminimizer_set(minimizer, &F, initial_guess, 0.0, 1.0e-8);

    // Öffne eine Datei zum Speichern der Datenpunkte
    FILE *dataFile = fopen("data_points.dat", "w");
    if (dataFile == NULL) {
        fprintf(stderr, "Error creating data file.\n");
        exit(EXIT_FAILURE);
    }

    // Berechne und speichere Datenpunkte für die Funktion und die Riemann p-Energie
    for (int i = 0; i < NUM_POINTS; ++i) {
        double u = -1.0 + i * 2.0 / (NUM_POINTS - 1);
        double u_prime = gsl_deriv_central(p_energy, u, 1e-8, p);
        fprintf(dataFile, "%lf %lf %lf\n", u, p_energy(u, u_prime, p), u_prime);
    }

    fclose(dataFile);

    // Öffne eine Datei zum Speichern der berechneten Minima
    FILE *minimaFile = fopen("minima.dat", "w");
    if (minimaFile == NULL) {
        fprintf(stderr, "Error creating minima file.\n");
        exit(EXIT_FAILURE);
    }

    // Durchführe die Minimierung
    int status;
    int iter = 0;
    do {
        iter++;
        status = gsl_min_fminimizer_iterate(minimizer);
        double minimum = gsl_min_fminimizer_x_minimum(minimizer);
        double lower_bound = gsl_min_fminimizer_x_lower(minimizer);
        double upper_bound = gsl_min_fminimizer_x_upper(minimizer);
        status = gsl_min_test_interval(lower_bound, upper_bound, 0.0, 1.0e-8);

        // Speichere die gefundenen Minima
        fprintf(minimaFile, "%lf %lf\n", minimum, p_energy(minimum, gsl_deriv_central(lagrange_equation, minimum, 1e-8, p), p));

    } while (status == GSL_CONTINUE && iter < 1000);

    fclose(minimaFile);

    // Freigabe des Speichers
    gsl_min_fminimizer_free(minimizer);

    // Erstelle GNUplot-Skript
    FILE *scriptFile = fopen("plotScript.gnu", "w");
    if (scriptFile == NULL) {
        fprintf(stderr, "Error creating plot script file.\n");
        exit(EXIT_FAILURE);
    }

    fprintf(scriptFile, "set terminal png\n");
    fprintf(scriptFile, "set output 'riemann_p_energy_plot.png'\n");
    fprintf(scriptFile, "plot 'data_points.dat' using 1:2 with lines title 'Riemann p-Energy', \
                        'data_points.dat' using 1:3 with lines title 'Derivative', \
                        'minima.dat' with points pointtype 7 title 'Minima'\n");

    fclose(scriptFile);

    // Ausführung von GNUplot
    system("gnuplot plotScript.gnu");

    // Entfernen temporärer Dateien
    remove("data_points.dat");
    remove("minima.dat");
    remove("plotScript.gnu");

    return 0;
}

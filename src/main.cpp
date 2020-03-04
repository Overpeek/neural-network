#include "helper.hpp"
#include "layer.hpp"

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>



int main() {
	std::mt19937 random = std::mt19937();
	std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);

	auto layer1 = Layer<2, 8>();
	auto layer2 = Layer<8, 1>();

	size_t i = 0;
	for (;;/*size_t i = 0; i < 1000000; i++*/) {
		double x = distribution(random);
		double y = distribution(random);
		arma::dvec2 layer1_input = { x, y };
		arma::dvec2 layer1_error;
		arma::dvec8 layer2_error;
		arma::dvec  layer2_expected = { 0.2*x + 0.6*y + 0.5 };

		// feed forward
		layer1.feedforward(layer1_input);
		layer2.feedforward(layer1.get_layer());
		
		// error
		arma::dvec error = arma::pow(layer2_expected - layer2.get_layer(), 1.0);

		// feed backward
		layer2.feedbackward(layer1.get_layer(), layer2_error, error);
		layer1.feedbackward(layer1_input, layer1_error, layer2_error);
		
		// every 50000 trainings
		const size_t progress_report_interval = 50000;
		if (i % progress_report_interval == progress_report_interval - 1) {
			std::cout << "progress report " << i << std::endl;
			std::cout << "cost:\n" << layer1.get_cost() << std::endl;
		}

		// every 1000 trainings
		const size_t apply_interval = 1000;
		if (i % apply_interval == apply_interval - 1) {
			layer1.apply(0.5);
			layer2.apply(0.5);
		}

		i++;
	}

	return 0;
}
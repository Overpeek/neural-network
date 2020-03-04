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

/*
// 3 inputs, 2 outputs
arma::dmat::fixed<2, 3> network;
arma::dmat::fixed<2, 3> network_error_batch;
double cost;

void feedforward(const arma::dvec::fixed<3>& input_layer, arma::dvec::fixed<2>& output_layer) {
	// feed forward
	output_layer = network * input_layer;
}

// returns error of input_layer and modifies weights between input_layer and output_layer
arma::dvec::fixed<3> feedbackward(const arma::dvec::fixed<3>& input_layer, arma::dvec::fixed<2>& output_layer, arma::dvec::fixed<2>& expected) {
	// error function
	arma::dvec::fixed<2> output_layer_error_linear = expected - output_layer;
	// arma::dvec::fixed<2> output_layer_error = arma::pow(output_layer_error_linear, 2.0) % arma::sign(output_layer_error_linear);
	arma::dvec::fixed<2> error = arma::pow(output_layer_error_linear, 3.0); // power 3 of error
	cost += arma::dot(error, error);
	
	const double learn_rate = 0.1;
	error *= learn_rate;

	// weight errors
	arma::dmat::fixed<2, 3> replicate_input_layer;
	for (size_t i = 0; i < network.n_rows; i++) {
		replicate_input_layer.row(i) = error.at(i) * input_layer.as_row();
	}

	// weight adjusting happens only when applied, it is stored until that
	auto adjust = (replicate_input_layer % network);
	network_error_batch += adjust;

	// return error of the input_layer
	return network.t() * error;
}

void operate(double x, double y) {
	// input and expected output
	arma::dvec::fixed<3> input_layer = { x, y, 1.0 };
	arma::dvec::fixed<2> output_layer;
	arma::dvec::fixed<2> expected_output_layer = { x, y };

	// feedforward pass
	feedforward(input_layer, output_layer);

	// train
	feedbackward(input_layer, output_layer, expected_output_layer);
}

void reset_error_batch() {
	network_error_batch = arma::dmat::fixed<2, 3>({
		//x,   y    bias
		{ 0.0, 0.0, 0.0 }, // output x
		{ 0.0, 0.0, 0.0 }, // output y
	});
	cost = 0.0;
}

int main() {
	std::mt19937 random = std::mt19937();
	std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);

	network = arma::dmat::fixed<2, 3>({
		//        x,                    y                     bias
		{ distribution(random), distribution(random), distribution(random) }, // output x
		{ distribution(random), distribution(random), distribution(random) }, // output y
	});
	reset_error_batch();

	for (size_t i = 0; i < 1000000; i++) {
		double x = distribution(random);
		double y = distribution(random);
		operate(x, y);

		if (i % 1000 == 0) { // every 1000 batches
			std::cout << "network:" << network << std::endl;
			std::cout << "cost   :" << cost << std::endl;

			network += network_error_batch * 0.001;
			reset_error_batch();
		}
	}

	std::cout << "network:" << network << std::endl;
	getchar();
	
	return 0;
}
*/
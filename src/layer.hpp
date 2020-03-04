void relu(double& value) {
	value = std::max(0.0, value);
}

void d_relu(double& value) {
	value = (value > 0.0 ? 1.0 : 0.0);
}

template<int input_size, int output_size>
class Layer {
private:
	// 3 inputs, 2 outputs
	arma::dmat::fixed<output_size, input_size> layer_weight;
	arma::dmat::fixed<output_size, input_size> layer_weight_error_batch;

	arma::dvec::fixed<output_size> layer_bias;
	arma::dvec::fixed<output_size> layer_bias_error_batch;

	// this holds the output
	arma::dvec::fixed<output_size> layer;
	arma::dvec::fixed<output_size> layer_error;

	double cost;
	size_t learn_calls;

public:
	Layer() {
		arma::arma_rng::set_seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		layer_weight.randu();
		layer_bias.randu();

		double weight_scale = 1.0 / (output_size * input_size);
		layer_weight *= weight_scale;
		layer_bias *= weight_scale;

		layer_weight_error_batch.fill(0.0);
		layer_bias_error_batch.fill(0.0);
		cost = 0.0;
		learn_calls = 0;
	}

	~Layer() {

	}

	void feedforward(const arma::dvec::fixed<input_size>& input_layer) {
		// feed forward
		layer = layer_weight * input_layer + layer_bias;
		// layer.for_each(relu);
	}

	// returns error of input_layer and modifies weights between input_layer and output_layer
	void feedbackward(const arma::dvec::fixed<input_size>& input_layer, arma::dvec::fixed<input_size>& input_layer_errors, const arma::dvec::fixed<output_size>& output_layer_error) {
		learn_calls++;
		
		// error function
		cost += arma::dot(output_layer_error, output_layer_error);

		// weight/bias errors
		arma::dmat::fixed<output_size, input_size> replicate_input_layer;
		for (size_t i = 0; i < output_size; i++) {
			replicate_input_layer.row(i) = output_layer_error.at(i) * input_layer.as_row();
		}

		// weight/bias adjusting happens only when applied, it is stored until that
		auto adjust_weight = (replicate_input_layer % layer_weight);
		auto adjust_bias = (output_layer_error % layer_bias);
		layer_weight_error_batch += adjust_weight;
		layer_bias_error_batch += adjust_bias;

		// return error of the input_layer
		input_layer_errors = layer_weight.t() * output_layer_error;
	}

	void apply(double learn_rate = 0.01) {
		// double final_rate = ;
		layer_weight += layer_weight_error_batch * learn_rate / learn_calls;
		layer_bias += layer_bias_error_batch * learn_rate / learn_calls;
		layer_weight_error_batch.fill(0.0);
		layer_bias_error_batch.fill(0.0);
		cost = 0.0;
		learn_calls = 0;
	}

public:
	const arma::dvec::fixed<output_size>& get_layer() const { return layer; }
	const double& get_cost() const { return cost; }
	arma::dvec::fixed<output_size>& get_layer_error() { return layer_error; }
	const arma::dmat::fixed<output_size, input_size>& get_weights() const { return layer_weight; }

};

/*
double relu(double value) {
	return std::max(0.0, value);
}

double d_relu(double value) {
	return value > 0.0 ? 1.0 : 0.0;
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
}*/
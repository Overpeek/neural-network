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
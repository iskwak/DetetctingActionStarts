"""GFlags for lstm."""
import gflags

gflags.DEFINE_integer("lstm_input_dim", 64, "Input feature dimensions.")
gflags.DEFINE_integer("lstm_hidden_dim", 64, "LSTM hidden dimensions.")
gflags.DEFINE_integer("lstm_num_layers", 2, "Number of lstm layers.")
gflags.DEFINE_boolean("lstm_bias", True, "Extra bias values for the lstm.")
gflags.DEFINE_float("lstm_dropout", 0.0, "LSTM dropout.")
gflags.DEFINE_boolean("lstm_bidirectional", False, "Bidirectional LSTM.")
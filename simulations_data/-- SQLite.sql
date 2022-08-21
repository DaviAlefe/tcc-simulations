CREATE TABLE IF NOT EXISTS Minified_Weights (start_neuron_idx INTEGER, end_neuron_idx INTEGER, connection_weight REAL)


INSERT INTO Minified_Weights
SELECT *
FROM (
	SELECT start_neuron_idx, end_neuron_idx,
	  LAST_VALUE(connection_weight) OVER (
	  PARTITION BY start_neuron_idx, end_neuron_idx
	  ORDER BY time ASC
	  RANGE BETWEEN UNBOUNDED PRECEDING AND 
	  UNBOUNDED FOLLOWING
	  ) AS connection_weight
	FROM Weights
)
GROUP BY start_neuron_idx, end_neuron_idx, connection_weight

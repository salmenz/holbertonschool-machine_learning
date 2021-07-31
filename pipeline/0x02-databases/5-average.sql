-- computes the score average of all records in the table second_table in your MySQL server

SELECT CAST(AVG(score) AS DECIMAL(10,2)) AS average FROM second_table;
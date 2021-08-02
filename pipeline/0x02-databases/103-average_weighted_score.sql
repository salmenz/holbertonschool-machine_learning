-- computes and store the average weighted score for a student.

DELIMITER //
CREATE PROCEDURE ComputeAverageWeightedScoreForUser(
	IN user_id_new INTEGER)
	BEGIN
		UPDATE users SET average_score=(
			SELECT SUM(score*weight)/SUM(weight) FROM corrections 
				JOIN projects ON corrections.project_id=projects.id
				WHERE user_id=user_id_new) WHERE id=user_id_new;
	END //
DELIMITER ;
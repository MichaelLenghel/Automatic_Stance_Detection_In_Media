DROP TABLE `topic`.`sentiments`;

CREATE TABLE `topic`.`sentiments`(
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `company` varchar(45) DEFAULT NULL,
  `topic` varchar(45) DEFAULT NULL,
  `year` year(4) DEFAULT NULL,
  `avg_topic_word_sentiment` float(5,4) DEFAULT NULL,
  `avg_sentiment` float(5,4) DEFAULT NULL,
  `avg_objectivity` float(5,4) DEFAULT NULL,
  `mode` float(2,1) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=latin1;

INSERT INTO `topic`.`sentiments`
(`company`,
`topic`,
`year`,
`avg_topic_word_sentiment`,
`avg_sentiment`,
`avg_objectivity`,
`mode`)
VALUES
('Daily Mail','abortion', 2019, 0.1130, 0.0491, 0.4155, 0.3),
('Daily Mail','brexit', 2019, 0.0544, 0.0812, 0.4175, 0.1),
('Daily Mail','business', 2019, 0.0893, 0.0622, 0.3873, -0.2),
('Daily Mail','christianity', 2019, 0.0595, 0.0431,0.4330, 0.1),
('Daily Mail','drugs', 2019, -0.427, -0.026, 0.4402, -0.1),
('Daily Mail','housing', 2019, 0.0555, 0.0815, 0.4244, 0.25),
('Daily Mail','medicine', 2019, 0.0662, 0.0453, 0.4626, 0.1),
('Daily Mail','online', 2019, 0.0503, 0.0942, 0.4665, 0.5),
('Daily Mail','sport', 2019, 0.2789, 0.1491, 0.4346, 0.2),
('Daily Mail','stocks', 2019, 0.0887, 0.0602, 0.3707, 0.1),
('Daily Mail','trump', 2019, 0.1495, 0.1296, 0.4210, 0.4),
('INDEPENDENT','abortion', 2019, 0.0714, 0.0795, 0.4550, 0.1),
('INDEPENDENT','brexit', 2019, 0.0148, 0.0597, 0.4965, 0.2),
('INDEPENDENT','business', 2019, 0.2056, 0.2176, 0.4965, 0.6),
('INDEPENDENT','China', 2019, 0.0354, 0.0671, 0.3707, 0.2),
('INDEPENDENT','christianity', 2019, 0.2112, 0.1721, 0.4411, 0.5),
('INDEPENDENT','drugs', 2019, 0.1400, 0.0282, 0.4459, 0.1),
('INDEPENDENT','housing', 2019, 0.0927, 0.1087, 0.4220, 0.0),
('INDEPENDENT','medicine', 2019, 0.0955, 0.0891, 0.4494, 0.2),
('INDEPENDENT','online', 2019, 0.1414, 0.1737, 0.4962, 0.5),
('INDEPENDENT','sports', 2019, 0.1095, 0.1156, 0.4585, 0.7),
('INDEPENDENT','trump', 2019, 0.0637, 0.0659, 0.4196, 0.1);
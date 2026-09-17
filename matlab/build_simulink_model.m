%% RetinaSight — Simulink District Rollout & Capacity Simulation Generator
% Smart India Hackathon 2026 | Problem Statement ID: 26038 (MathWorks)
% Generates the official Simulink architectural simulation model:
% 'retinasight_capacity_model.slx'

function mdl = build_simulink_model()
	mdl = 'retinasight_capacity_model';
	
	% Close if already open
	if bdIsLoaded(mdl)
		close_system(mdl, 0);
	end

	% Create fresh new model
	new_system(mdl);
	load_system(mdl);

	% Configure simulation parameters (30 days of rural screening)
	set_param(mdl, 'StopTime', '30');
	set_param(mdl, 'Solver', 'ode45');

	% 1. Patient Arrival Source (Constant: 180 patients/day at PHC cluster)
	add_block('simulink/Sources/Constant', [mdl, '/Daily_Patient_Intake'], ...
	          'Position', [60, 80, 160, 120], ...
	          'Value', '180');

	% 2. Quality Assurance Gate (Gain: 88% first-time pass rate, 12% retake)
	add_block('simulink/Math Operations/Gain', [mdl, '/Quality_Gate_Pass'], ...
	          'Position', [220, 80, 320, 120], ...
	          'Gain', '0.88');

	% 3. Cumulative Screened Integrator
	add_block('simulink/Continuous/Integrator', [mdl, '/Cumulative_Screened'], ...
	          'Position', [380, 80, 430, 120]);

	% 4. DR Positive Triage Split (Gain: 32% Grade 2-4 requiring ophthalmologist referral)
	add_block('simulink/Math Operations/Gain', [mdl, '/DR_Referral_Triage'], ...
	          'Position', [220, 200, 320, 240], ...
	          'Gain', '0.32');

	% 5. Tertiary Hospital Daily Review Capacity (Saturation: Max 45 consultations/day)
	add_block('simulink/Discontinuities/Saturation', [mdl, '/Hospital_Capacity_Limit'], ...
	          'Position', [380, 200, 460, 240], ...
	          'UpperLimit', '45', 'LowerLimit', '0');

	% 6. Cumulative Specialist Tele-Consultations
	add_block('simulink/Continuous/Integrator', [mdl, '/Cumulative_Tele_Consults'], ...
	          'Position', [520, 200, 570, 240]);

	% 7. Scope: Screening Throughput
	add_block('simulink/Sinks/Scope', [mdl, '/Screening_Throughput_Scope'], ...
	          'Position', [500, 80, 550, 120]);

	% 8. Scope: Hospital Referrals
	add_block('simulink/Sinks/Scope', [mdl, '/Specialist_Referrals_Scope'], ...
	          'Position', [630, 200, 680, 240]);

	% 9. To Workspace: Cumulative Screened
	add_block('simulink/Sinks/To Workspace', [mdl, '/ToWS_Screened'], ...
	          'Position', [500, 20, 580, 50], ...
	          'VariableName', 'total_screened', 'SaveFormat', 'Array');

	% 10. To Workspace: Cumulative Tele-Consultations
	add_block('simulink/Sinks/To Workspace', [mdl, '/ToWS_TeleConsults'], ...
	          'Position', [630, 270, 710, 300], ...
	          'VariableName', 'total_teleconsults', 'SaveFormat', 'Array');

	% Connect signal lines
	add_line(mdl, 'Daily_Patient_Intake/1', 'Quality_Gate_Pass/1');
	add_line(mdl, 'Quality_Gate_Pass/1', 'Cumulative_Screened/1');
	add_line(mdl, 'Quality_Gate_Pass/1', 'DR_Referral_Triage/1');
	add_line(mdl, 'Cumulative_Screened/1', 'Screening_Throughput_Scope/1');
	add_line(mdl, 'Cumulative_Screened/1', 'ToWS_Screened/1');

	add_line(mdl, 'DR_Referral_Triage/1', 'Hospital_Capacity_Limit/1');
	add_line(mdl, 'Hospital_Capacity_Limit/1', 'Cumulative_Tele_Consults/1');
	add_line(mdl, 'Cumulative_Tele_Consults/1', 'Specialist_Referrals_Scope/1');
	add_line(mdl, 'Cumulative_Tele_Consults/1', 'ToWS_TeleConsults/1');

	% Save system
	modelFilePath = fullfile(pwd, 'retinasight_capacity_model.slx');
	save_system(mdl, modelFilePath);
	close_system(mdl, 0);
	fprintf('[SUCCESS] Simulink model saved: %s\n', modelFilePath);
end

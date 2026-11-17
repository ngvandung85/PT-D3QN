% RUN_OBJECTIVE_TRAINING_CSV.m
% Read the supplied summary CSV; no embedded measurements.
% Open this script in MATLAB Editor and press Run (F5).
% Expected source: user_count_training_curves.csv, episodes 0..2999.
% Editable FIG files plus PDF/PNG are saved in objective_figures beside script.
% No toolbox required. Uses readtable, subplot, savefig (or saveas), print.
% This script has not been executed in MATLAB in the preparation environment.

% OPTIONAL: set a full CSV path here; leave empty for automatic lookup/dialog.
csvFile = ''; % Example: 'D:\results\user_count_training_curves.csv'
plotStride = 50; % 1 = all episodes; 50 = episodes 1,51,...,2951,3000.

scriptPath = mfilename('fullpath');
if isempty(scriptPath), baseDir = pwd; else, baseDir = fileparts(scriptPath); end
if isempty(csvFile)
    csvFile = fullfile(baseDir,'user_count_training_curves.csv');
    if exist(csvFile,'file')~=2
        csvFile = fullfile(pwd,'user_count_training_curves.csv');
    end
end
if exist(csvFile,'file')~=2
    [fileName,folderName] = uigetfile('*.csv','Select user_count_training_curves.csv');
    if isequal(fileName,0), disp('Cancelled: no CSV selected.'); return; end
    csvFile = fullfile(folderName,fileName);
end
assert(isscalar(plotStride) && plotStride>=1 && plotStride==floor(plotStride), ...
    'plotStride must be a positive integer.');
T = readtable(csvFile);
required = {'n_user','method','episode','objective_return_mean', ...
    'objective_return_std','objective_return_ci95_half_width','n_replicates'};
missing = required(~ismember(required,T.Properties.VariableNames));
if ~isempty(missing)
    error('CSV missing required columns: %s',strjoin(missing,', '));
end
usersToRead = [5 15 30 45 60];
methodsToRead = {'EUT_D3QN','PT_D3QN'};
objectiveData = []; % U, method index, one-based episode, mean, CI half-width.
for i = 1:numel(usersToRead)
    for j = 1:numel(methodsToRead)
        selected = T.n_user==usersToRead(i) & strcmp(T.method,methodsToRead{j});
        S = sortrows(T(selected,:),'episode');
        if height(S)~=3000 || ~isequal(S.episode(:),(0:2999)')
            error('U=%d, %s: expected 3000 unique episodes numbered 0..2999.', ...
                usersToRead(i),methodsToRead{j});
        end
        if any(S.n_replicates~=10)
            error('U=%d, %s: expected ten replicates per episode.', ...
                usersToRead(i),methodsToRead{j});
        end
        V = [S.objective_return_mean,S.objective_return_std, ...
             S.objective_return_ci95_half_width];
        if any(~isfinite(V(:))) || any(S.objective_return_std<0) || ...
                any(S.objective_return_ci95_half_width<0)
            error('Nonfinite values or negative standard deviations/CI half-widths.');
        end
        % Stored pointwise Student-t intervals, df=9; no Statistics Toolbox.
        expectedCI = 2.262157162740992*S.objective_return_std/sqrt(10);
        if max(abs(expectedCI-S.objective_return_ci95_half_width))>1e-8
            error('Stored CI does not match a 95%% t interval across ten seeds.');
        end
        idx = unique([1:plotStride:height(S),height(S)]);
        D = S(idx,:); n = height(D);
        objectiveData = [objectiveData; ...
            usersToRead(i)*ones(n,1),j*ones(n,1),D.episode+1, ...
            D.objective_return_mean,D.objective_return_ci95_half_width]; %#ok<AGROW>
    end
end
fprintf('Loaded and validated data from: %s\n',csvFile);
% Means and stored CI are sampled, not temporally smoothed or recomputed.
% Episode reward sums are undiscounted; reward scales differ between methods.

users = [5 15 30 45 60];
methodNames = {'EUT-D3QN','PT-D3QN'};
colors = [0 0.35 0.65; 0.80 0.25 0.08];
scriptPath = mfilename('fullpath');
if isempty(scriptPath), baseDir = pwd; else, baseDir = fileparts(scriptPath); end
outDir = fullfile(baseDir,'objective_figures');
if ~exist(outDir,'dir')
    [ok,msg] = mkdir(outDir);
    if ~ok, error('Cannot create output folder: %s',msg); end
end
ylimits = zeros(2,2);
for j = 1:2
    D = objectiveData(objectiveData(:,2)==j,:);
    lo = min(D(:,4)-D(:,5)); hi = max(D(:,4)+D(:,5));
    pad = 0.06*(hi-lo); ylimits(j,:) = [lo-pad hi+pad];
end
% First export a combined figure, then five per-population figures.
for figureIndex = 0:5
    if figureIndex==0
        selectedRows = 1:5;
        nRows = 5;
        figName = 'Fig_Objective_Training';
        figHeight = 26;
    else
        selectedRows = figureIndex;
        nRows = 1;
        figName = sprintf('Fig_Objective_U%03d',users(figureIndex));
        figHeight = 8;
    end
    f = figure('Color','w','Units','centimeters', ...
        'Position',[1 1 19 figHeight],'Name',figName,'NumberTitle','off');
    for rowIndex = 1:numel(selectedRows)
        i = selectedRows(rowIndex);
        for j = 1:2
            ax = subplot(nRows,2,(rowIndex-1)*2+j,'Parent',f);
            D = objectiveData(objectiveData(:,1)==users(i) & objectiveData(:,2)==j,:);
            x=D(:,3); y=D(:,4); h=D(:,5);
            hold(ax,'on');
            fill([x;flipud(x)],[y-h;flipud(y+h)], ...
                0.84*[1 1 1]+0.16*colors(j,:), 'EdgeColor','none','Parent',ax);
            plot(ax,x,y,'Color',colors(j,:),'LineWidth',1.1);
            set(ax,'FontName','Times New Roman','FontSize',10, ...
                'Box','on','TickDir','out','XLim',[1 3000], ...
                'XTick',[1 1000 2000 3000],'YLim',ylimits(j,:),'Layer','top');
            grid(ax,'on');
            title(ax,sprintf('(%c) %s | U = %d', ...
                char(97+(i-1)*2+j-1),methodNames{j},users(i)), ...
                'FontWeight','normal');
            if rowIndex==nRows, xlabel(ax,'Training episode'); end
            if j==1, ylabel(ax,'Episode reward sum'); end
        end
    end
    drawnow;
    figPath=fullfile(outDir,[figName '.fig']);
    % Save FIG before attempting optional PDF/PNG exports.
    if exist('savefig','file')~=0
        savefig(f,figPath);
    else
        saveas(f,figPath,'fig');
    end
    assert(exist(figPath,'file')==2,'FIG was not saved. Check folder permissions.');
    fprintf('Created editable figure: %s\n',figPath);
    set(f,'PaperUnits','centimeters','PaperSize',[19 figHeight], ...
        'PaperPosition',[0 0 19 figHeight],'PaperPositionMode','manual');
    try
        print(f,fullfile(outDir,[figName '.pdf']),'-dpdf','-painters');
        print(f,fullfile(outDir,[figName '.png']),'-dpng','-r300');
    catch exportError
        warning('FIG saved successfully; PDF/PNG export failed: %s',exportError.message);
    end
end
fprintf('\nDone. Open the combined figure with:\n');
fprintf('openfig(''%s'')\n',fullfile(outDir,'Fig_Objective_Training.fig'));

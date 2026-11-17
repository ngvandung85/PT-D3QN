function make_training_figure(dataDir, outDir)
%MAKE_TRAINING_FIGURE  Hinh 3: loi ich vat ly khi huan luyen, nam chinh sach.
%
%   make_training_figure                    % CSV o thu muc hien tai
%   make_training_figure(dataDir)           % chi dinh thu muc chua CSV
%   make_training_figure(dataDir, outDir)   % chi dinh ca thu muc xuat
%
% Dau vao:
%   user_count_training_curves.csv
%       cot: n_user, method, episode,
%            physical_utility_return_mean,
%            physical_utility_return_ci95_half_width,
%            n_replicates  (va cac cot objective_* khong dung o hinh nay)
%       75000 dong = 5 user count x 5 chinh sach x 3000 episode
%
% Dau ra trong figures/:
%   Fig_Training_PhysUtil.fig   de chinh sua bang tay
%   Fig_Training_PhysUtil.pdf   vector, chen thang vao LaTeX
%   Fig_Training_PhysUtil.png   xem nhanh
%
% Hinh ve lai Hinh 3 theo dung caption cua ban thao:
%   - nam panel, moi panel mot user count
%   - noi cac episode 1, 51, ..., 2951 va 3000, KHONG lam tron
%   - dai mo la khoang tin cay Student-t 95% theo diem qua muoi seed
%   - episode hien thi danh so tu 1: diem e+1 ung voi chi so e trong (26)
%
% Ba sua doi theo yeu cau cua Reviewer 4 muc (b):
%   1. Co chu truc tang len 10 va chu legend len 9 (ban cu dung 6-7).
%   2. Moi chinh sach phan biet bang MAU, KIEU NET va DAU MARKER, nen
%      hinh con doc duoc khi in den trang. Hai nhanh chinh (EUT-D3QN,
%      PT-D3QN) giu net lien va phan biet nhau bang marker; ba nhanh
%      ablation dung net gach hoac cham.
%   3. Legend dat o o thu sau cua luoi 2x3 nen khong che duong, va cac
%      panel rong hon.
%
% Hai co de chinh tay:
%   CI_MODE  = 'main' (mac dinh) chi ve dai tin cay cho EUT-D3QN va
%              PT-D3QN, tuc cap duoc so sanh trong bai. Nam dai chong
%              nhau lam nen bi duc va chinh la dieu Reviewer 4 phan nan.
%              'all'  ve du nam dai; 'none' bo han.
%              DUNG 'main' THI PHAI SUA CAPTION: xem cuoi file.
%   SHARE_Y  = false  neu muon moi panel tu chia truc y
%
% Yeu cau MATLAB R2020a tro len. KHONG can Statistics and Machine
% Learning Toolbox: khoang tin cay da co san trong CSV.

if nargin < 1 || isempty(dataDir), dataDir = pwd; end
if nargin < 2 || isempty(outDir),  outDir  = fullfile(pwd, 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end

CI_MODE  = 'main';                   % 'main' | 'all' | 'none'
MAIN     = ["EUT_D3QN", "PT_D3QN"];  % cap duoc so sanh trong bai
SHARE_Y  = true;
FONT     = resolve_font('DejaVu Serif');
FS       = 10;                       % chu truc; ban cu dung 9
FS_LEG   = 9;                        % chu legend; ban cu dung 6-7
EVERY    = 6;                        % dat marker moi 6 diem da lay thua
YPAD     = 0.03;                     % noi truc y theo ti le

% Nam chinh sach: mau, kieu net, marker. Hai nhanh chinh xep cuoi de ve
% tren cung, khong bi ba nhanh ablation che.
% Nam mau tach xa nhau ve sac. Ba bien the EUT KHONG cung mot ho xanh:
% ba sac xanh canh nhau lam EUT-DDQN gan nhu bien mat. Hai nhanh chinh
% giu dung mau quy uoc cua du an (#354c7c va #b74535).
specs = {
    "EUT_DQN",         [0.5490 0.5490 0.5490], ':',  '^', 'EUT-DQN'
    "EUT_DDQN",        [0.4824 0.3529 0.6510], '-.', 'd', 'EUT-DDQN'
    "EUT_Dueling_DQN", [0.3098 0.6157 0.4118], '--', 's', 'EUT-Dueling-DQN'
    "EUT_D3QN",        [0.2078 0.2980 0.4863], '-',  'o', 'EUT-D3QN'
    "PT_D3QN",         [0.7176 0.2706 0.2078], '-',  'v', 'PT-D3QN'
};

T = readtable(fullfile(dataDir, 'user_count_training_curves.csv'));
T.method = string(T.method);
validate_inputs(T, specs);

% Episode lay thua dung nhu caption: chi so 0-based 0, 50, ..., 2950 va 2999.
keep  = unique([0:50:2950, 2999]);
users = unique(T.n_user);

f = figure('Name', 'Fig_Training_PhysUtil', 'Color', 'w', ...
           'Units', 'centimeters', 'Position', [1 1 17.0 9.8]);
tl = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Gioi han y dung chung de so sanh giua cac panel la trung thuc.
if SHARE_Y
    [ylo, yhi] = global_ylim(T, keep, CI_MODE, MAIN, YPAD);
end

ax    = gobjects(1, numel(users));
hLeg  = gobjects(1, size(specs, 1));
for k = 1:numel(users)
    ax(k) = nexttile(tl);
    hold(ax(k), 'on');

    for j = 1:size(specs, 1)
        [x, mu, hw] = series(T, users(k), specs{j,1}, keep);
        c = specs{j,2};

        if draw_ci(CI_MODE, MAIN, specs{j,1})
            fill(ax(k), [x; flipud(x)], [mu-hw; flipud(mu+hw)], c, ...
                 'FaceAlpha', 0.16, 'EdgeColor', 'none', ...
                 'HandleVisibility', 'off');
        end

        h = plot(ax(k), x, mu, specs{j,3}, 'Color', c, 'LineWidth', 1.5, ...
                 'Marker', specs{j,4}, 'MarkerSize', 4.2, ...
                 'MarkerFaceColor', c, 'MarkerEdgeColor', 'none', ...
                 'MarkerIndices', 1:EVERY:numel(x), ...
                 'DisplayName', specs{j,5});
        if k == 1, hLeg(j) = h; end
    end

    title(ax(k), sprintf('%d users', users(k)), ...
          'FontName', FONT, 'FontSize', FS, 'FontWeight', 'normal');
    xlim(ax(k), [0 3000]);
    set(ax(k), 'XTick', 0:1000:3000);
    if SHARE_Y, ylim(ax(k), [ylo yhi]); end
    style_axes(ax(k), FONT, FS);
end

% Nhan truc dat o muc layout nen khong lap lai tren tung panel.
xlabel(tl, 'Training episode', 'FontName', FONT, 'FontSize', FS);
ylabel(tl, 'Physical-utility return', 'FontName', FONT, 'FontSize', FS);

% O thu sau: chi dung de chua legend.
axLeg = nexttile(tl);
axis(axLeg, 'off');
lg = legend(axLeg, hLeg, 'FontName', FONT, 'FontSize', FS_LEG, ...
            'Box', 'off', 'Location', 'none');
lg.Units    = 'normalized';
lg.Position = axLeg.Position;

save_figure(f, outDir, 'Fig_Training_PhysUtil');

report(T, keep);

end


% =====================================================================
function [x, mu, hw] = series(T, nUser, method, keep)
%SERIES  Chuoi da lay thua cho mot (user count, chinh sach).
sub = T(T.n_user == nUser & T.method == method, :);
sub = sortrows(sub, 'episode');
sel = ismember(sub.episode, keep);
sub = sub(sel, :);
assert(height(sub) == numel(keep), ...
       'Thieu episode o U=%d, %s: co %d can %d', ...
       nUser, method, height(sub), numel(keep));

% Episode hien thi danh so tu 1: diem e+1 ung voi chi so e trong (26).
x  = sub.episode(:) + 1;
mu = sub.physical_utility_return_mean(:);
hw = sub.physical_utility_return_ci95_half_width(:);
end


% =====================================================================
function tf = draw_ci(mode, mainMethods, method)
%DRAW_CI  Co ve dai tin cay cho chinh sach nay khong.
switch lower(mode)
    case 'all',  tf = true;
    case 'none', tf = false;
    case 'main', tf = any(mainMethods == method);
    otherwise,   error('CI_MODE phai la ''main'', ''all'' hoac ''none''');
end
end


% =====================================================================
function [ylo, yhi] = global_ylim(T, keep, mode, mainMethods, pad)
%GLOBAL_YLIM  Gioi han y dung chung cho moi panel.
%   Luon bao duong trung binh cua ca nam chinh sach, va bao them dai tin
%   cay cua nhung chinh sach thuc su duoc ve dai.
sel = ismember(T.episode, keep);
S   = T(sel, :);
lo  = min(S.physical_utility_return_mean);
hi  = max(S.physical_utility_return_mean);

for m = unique(S.method)'
    if ~draw_ci(mode, mainMethods, m), continue; end
    r  = S(S.method == m, :);
    lo = min(lo, min(r.physical_utility_return_mean - r.physical_utility_return_ci95_half_width));
    hi = max(hi, max(r.physical_utility_return_mean + r.physical_utility_return_ci95_half_width));
end

span = hi - lo;
ylo  = lo - pad * span;
yhi  = hi + pad * span;
end


% =====================================================================
function report(T, keep)
%REPORT  In cac moc kiem chung de doi chieu voi Muc 6.2.
fprintf('\n--- Kiem chung (phai khop so lieu trong Muc 6.2) ---\n');
last = max(keep);                       % chi so 0-based cua episode cuoi
E    = T(T.episode == last, :);

users = unique(T.n_user);
nUp   = 0;
for k = 1:numel(users)
    pt  = E.physical_utility_return_mean(E.n_user == users(k) & E.method == "PT_D3QN");
    eut = E.physical_utility_return_mean(E.n_user == users(k) & E.method == "EUT_D3QN");
    if pt > eut, nUp = nUp + 1; end
    fprintf('  U=%2d  episode %d: PT %.2f vs EUT %.2f  (%s)\n', ...
            users(k), last + 1, pt, eut, ternary(pt > eut, 'PT cao hon', 'PT thap hon'));
end

% Ban thao neu ro: PT cao hon o bon user count, va o 30 users la 44.51
% so voi 46.45. Hai moc nay phai dung.
fprintf('  PT-D3QN cao hon EUT-D3QN o %d trong %d user count ', nUp, numel(users));
fprintf('(ban thao neu: bon)\n');

pt30  = E.physical_utility_return_mean(E.n_user == 30 & E.method == "PT_D3QN");
eut30 = E.physical_utility_return_mean(E.n_user == 30 & E.method == "EUT_D3QN");
assert(abs(pt30 - 44.51) < 0.005 && abs(eut30 - 46.45) < 0.005, ...
       ['Moc 30 users khong khop ban thao: duoc %.4f va %.4f, ' ...
        'can 44.51 va 46.45'], pt30, eut30);
fprintf('  Moc 30 users: DAT (%.2f vs %.2f)\n', pt30, eut30);
fprintf('  So diem moi duong: %d (episode %d, %d, ..., %d, %d)\n', ...
        numel(keep), keep(1)+1, keep(2)+1, keep(end-1)+1, keep(end)+1);
end


% =====================================================================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


% =====================================================================
function style_axes(ax, fontName, fontSize)
box(ax, 'off');
grid(ax, 'on');
ax.GridAlpha  = 0.20;
ax.FontName   = fontName;
ax.FontSize   = fontSize;
ax.LineWidth  = 0.8;
ax.TickDir    = 'out';
ax.XColor     = [0.15 0.15 0.15];
ax.YColor     = [0.15 0.15 0.15];
end


% =====================================================================
function name = resolve_font(preferred)
available = listfonts;
if any(strcmp(available, preferred))
    name = preferred;
    return
end
fallbacks = {'Times New Roman', 'Nimbus Roman', 'Serif'};
name = 'Times New Roman';
for i = 1:numel(fallbacks)
    if any(strcmp(available, fallbacks{i}))
        name = fallbacks{i};
        break
    end
end
warning(['Khong tim thay phong "%s". Dung "%s" thay the. ' ...
         'Hinh se khac cac hinh khac o phong chu.'], preferred, name);
end


% =====================================================================
function validate_inputs(T, specs)
need = {'n_user', 'method', 'episode', ...
        'physical_utility_return_mean', ...
        'physical_utility_return_ci95_half_width', 'n_replicates'};
for i = 1:numel(need)
    assert(ismember(need{i}, T.Properties.VariableNames), ...
           'Thieu cot "%s" trong CSV', need{i});
end
assert(height(T) == 75000, ...
       'Bang phai co 75000 dong, dang co %d', height(T));
assert(numel(unique(T.n_user)) == 5, 'Can dung 5 user count');
assert(numel(unique(T.episode)) == 3000, 'Can dung 3000 episode');
for j = 1:size(specs, 1)
    assert(any(T.method == specs{j,1}), ...
           'Thieu chinh sach %s trong CSV', specs{j,1});
end
assert(numel(unique(T.method)) == size(specs, 1), ...
       'CSV co %d chinh sach, bang mau khai bao %d', ...
       numel(unique(T.method)), size(specs, 1));
assert(all(T.n_replicates == 10), ...
       'Moi dong phai dung 10 replicate');
assert(all(T.physical_utility_return_ci95_half_width >= 0), ...
       'Nua do rong khoang tin cay khong duoc am');
fprintf(['Kiem tra dau vao: DAT (75000 dong, 5 user count, 5 chinh sach, ' ...
         '3000 episode, 10 replicate)\n']);
end


% =====================================================================
function save_figure(fh, outDir, name)
%SAVE_FIGURE  Luu .fig, .pdf va .png, sau do mo lai .fig de kiem tra.
figPath = fullfile(outDir, [name '.fig']);
pdfPath = fullfile(outDir, [name '.pdf']);
pngPath = fullfile(outDir, [name '.png']);

savefig(fh, figPath);
try
    exportgraphics(fh, pdfPath, 'ContentType', 'vector', ...
                   'BackgroundColor', 'white');
    exportgraphics(fh, pngPath, 'Resolution', 220, ...
                   'BackgroundColor', 'white');
catch
    set(fh, 'PaperPositionMode', 'auto');
    print(fh, pdfPath, '-dpdf', '-painters');
    print(fh, pngPath, '-dpng', '-r220');
end
close(fh);

chk = openfig(figPath, 'invisible');
assert(isgraphics(chk, 'figure'), 'FIG khong hop le: %s', figPath);
close(chk);
fprintf('  da luu va kiem tra: %s (.fig, .pdf, .png)\n', name);
end


% =====================================================================
% CAPTION
% =====================================================================
% Voi CI_MODE = 'main' (mac dinh), caption cua Hinh 3 phai noi ro dai
% tin cay chi ve cho hai nhanh chinh. Ban de nghi:
%
%   Figure 3. Training physical-utility returns for the five learning
%   policies across user counts. Lines connect unsmoothed episode means
%   at episodes 1, 51, ..., 2951 and 3000. Policies are distinguished by
%   colour, line style and marker so that the panels remain legible
%   without colour. Shaded regions indicate pointwise 95% Student-t
%   confidence intervals across ten seeds for EUT-D3QN and PT-D3QN, the
%   pair compared in Table 3; intervals for the three remaining policies
%   are omitted for legibility and are available in the deposited
%   training-curve records. These are pointwise intervals, not
%   simultaneous confidence bands. Episodes are numbered from one here
%   and in the text; displayed episode e + 1 corresponds to the
%   zero-based training index e used in (26).
%
% Voi CI_MODE = 'all', giu nguyen caption cu: no da dung.

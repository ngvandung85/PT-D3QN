function make_energy_figure(dataDir, outDir)
%MAKE_ENERGY_FIGURE  Hinh nang luong con lai theo slot tai U = 60.
%
%   make_energy_figure                    % CSV o thu muc hien tai
%   make_energy_figure(dataDir)           % chi dinh thu muc chua CSV
%   make_energy_figure(dataDir, outDir)   % chi dinh ca thu muc xuat
%
% Dau vao:
%   energy_timeseries_u060.csv
%       cot: method, replicate_seed, slot, residual_energy_j
%       1600 dong = 2 phuong phap x 10 seed x 80 slot
%       Nang luong con lai = 6.0 J tru tieu thu tich luy trong tung episode,
%       lay trung binh qua 100 episode trong moi seed.
%
% Dau ra trong figures/:
%   Fig_Energy_U60.fig   de chinh sua bang tay
%   Fig_Energy_U60.pdf   vector, chen thang vao LaTeX
%   Fig_Energy_U60.png   xem nhanh
%
% Truc hoanh la SO SLOT DA HOAN THANH: diem 0 ung voi ngan sach ban dau
% 6 J, diem 60 ung voi luc chinh sach bao hoa tieu het ngan sach. Dinh
% nghia nay khop voi caption trong ban thao.
%
% Hinh ve dong thoi:
%   - duong trung binh qua 10 seed, kem dai tin cay Student-t 95%
%   - 10 duong manh cua tung seed, theo dung quy uoc bao cao gia tri
%     tung seed canh moi gia tri tong hop da neu trong Muc 5.3; day cung
%     la thu lam lo ra tinh hai dinh cua EUT-D3QN
%   - duong doc danh dau moc 60 slot hoan thanh
%
% Can duoi cua dai tin cay KHONG bi cat tai 0, dung nhu caption da neu.
%
% Yeu cau MATLAB R2020a tro len. KHONG can Statistics and Machine
% Learning Toolbox.

if nargin < 1 || isempty(dataDir), dataDir = pwd; end
if nargin < 2 || isempty(outDir),  outDir  = fullfile(pwd, 'figures'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end

COL_EUT  = [0.2078 0.2980 0.4863];   % #354c7c
COL_PT   = [0.7176 0.2706 0.2078];   % #b74535
FONT     = resolve_font('DejaVu Serif');
FS       = 13;
BUDGET   = 6.0;                      % 3 UAV x 2 J
MARK     = 60;                       % so slot hoan thanh khi can ngan sach
YLO      = -0.35;                    % de lo can duoi CI am (xem caption)

T = readtable(fullfile(dataDir, 'energy_timeseries_u060.csv'));
T.method = string(T.method);
validate_inputs(T);

f = figure('Name', 'Fig_Energy_U60', 'Color', 'w', ...
           'Units', 'centimeters', 'Position', [2 2 9.0 7.0]);
ax = axes(f);
hold(ax, 'on');

specs = {"EUT_D3QN", COL_EUT, 'EUT-D3QN'; "PT_D3QN", COL_PT, 'PT-D3QN'};
h = gobjects(1, 2);
for j = 1:2
    [slots, M] = seed_matrix(T, specs{j,1});     % [slot x seed]
    c  = specs{j,2};
    mu = mean(M, 2);
    hw = tquant(size(M,2)-1) * std(M, 0, 2) / sqrt(size(M,2));

    % Duong manh cho tung seed: giu bat dinh giua cac seed hien ro.
    plot(ax, slots, M, '-', 'Color', [c 0.22], 'LineWidth', 0.5, ...
         'HandleVisibility', 'off');

    fill(ax, [slots; flipud(slots)], [mu-hw; flipud(mu+hw)], c, ...
         'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    h(j) = plot(ax, slots, mu, '-', 'Color', c, 'LineWidth', 1.6, ...
                'DisplayName', specs{j,3});
end

yline(ax, 0, '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 0.6, ...
      'HandleVisibility', 'off');
xline(ax, MARK, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, ...
      'HandleVisibility', 'off');
text(ax, MARK - 1.4, 2.0, sprintf('%d completed slots', MARK), ...
     'Rotation', 90, 'FontName', FONT, 'FontSize', 7.5, ...
     'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'left', ...
     'VerticalAlignment', 'bottom', 'BackgroundColor', 'w', 'Margin', 0.5);

xlabel(ax, 'Completed slots');
ylabel(ax, 'Total residual radio energy (J)');
xlim(ax, [0 80]);
ylim(ax, [YLO BUDGET]);
set(ax, 'XTick', 0:10:80, 'YTick', 0:1:BUDGET);
legend(ax, h, 'Location', 'southwest', 'Box', 'off', 'FontSize', 7.5);
style_axes(ax, FONT, FS);

save_figure(f, outDir, 'Fig_Energy_U60');

report(T);

end


% =====================================================================
function [slots, M] = seed_matrix(T, method)
%SEED_MATRIX  Ma tran [slot x seed] cua nang luong con lai.
sub   = T(T.method == method, :);
seeds = unique(sub.replicate_seed);
slots = unique(sub.slot);
M     = nan(numel(slots), numel(seeds));
for s = 1:numel(seeds)
    r = sortrows(sub(sub.replicate_seed == seeds(s), :), 'slot');
    assert(isequal(r.slot, slots), 'Thieu slot o seed %d', seeds(s));
    M(:, s) = r.residual_energy_j;
end

% Doi sang truc "so slot da hoan thanh": them diem goc (0 slot, 6 J)
% de khop voi caption cua hinh.
slots = [0; slots(:) + 1];
M     = [repmat(6.0, 1, numel(seeds)); M];
end


% =====================================================================
function report(T)
%REPORT  In cac moc kiem chung de doi chieu voi phan van ban.
fprintf('\n--- Kiem chung (phai khop so lieu trong Muc 6.4) ---\n');
cond = struct();
for method = ["PT_D3QN", "EUT_D3QN"]
    [slots, M] = seed_matrix(T, method);
    mu   = mean(M, 2);
    i60  = find(slots == 60, 1);
    exh  = M(end, :) < 0.05;                 % seed can ngan sach
    cond.(method) = mean(M(:, exh), 2);      % quy dao rieng cac seed can
    fprintf(['  %-9s sau 60 slot %.4f J | cuoi episode %.4f J | ' ...
             'can: %d/%d seed\n'], ...
            method, mu(i60), mu(end), sum(exh), size(M,2));
end

% Luan diem trung tam: toc do tieu hao giong nhau, chi khac tan suat.
a = cond.PT_D3QN;  b = cond.EUT_D3QN;
k = 1:56;
rPT  = -polyfit(k(:), a(k), 1);
rEUT = -polyfit(k(:), b(k), 1);
fprintf(['  Chi xet cac seed can ngan sach: lech toi da %.4f J | ' ...
         'toc do %.5f vs %.5f J/slot\n'], max(abs(a-b)), rPT(1), rEUT(1));
end


% =====================================================================
function style_axes(ax, fontName, fontSize)
box(ax, 'off');
grid(ax, 'on');
ax.GridAlpha  = 0.20;
ax.FontName   = fontName;
ax.FontSize   = fontSize;
ax.LineWidth  = 1;
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
function q = tquant(df)
%TQUANT  Phan vi 0.975 cua Student-t. Khong can Statistics Toolbox.
dfs = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 40 60 120];
qs  = [12.706205 4.302653 3.182446 2.776445 2.570582 2.446912 2.364624 ...
       2.306004 2.262157 2.228139 2.200985 2.178813 2.160369 2.144787 ...
       2.131450 2.119905 2.109816 2.100922 2.093024 2.085963 2.059539 ...
       2.042272 2.021075 2.000298 1.979930];
idx = find(dfs == df, 1);
assert(~isempty(idx), ...
       ['Chua co phan vi t cho df = %d. Bo sung vao bang trong tquant() ' ...
        'bang gia tri 0.975 hai phia.'], df);
q = qs(idx);
end


% =====================================================================
function validate_inputs(T)
assert(height(T) == 1600, ...
       'Bang phai co 1600 dong, dang co %d', height(T));
for method = ["PT_D3QN", "EUT_D3QN"]
    sub = T(T.method == method, :);
    assert(numel(unique(sub.replicate_seed)) == 10, ...
           'Can dung 10 seed cho %s', method);
    assert(numel(unique(sub.slot)) == 80, ...
           'Can dung 80 slot cho %s', method);
end
assert(all(T.residual_energy_j >= -1e-9 & T.residual_energy_j <= 6 + 1e-9), ...
       'Nang luong con lai nam ngoai khoang [0, 6] J');
fprintf('Kiem tra dau vao: DAT (1600 dong, 10 seed, 80 slot, 0 <= E <= 6 J)\n');
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

;;; nngrep.el --- Semantic search with nngrep  -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "29.1") (transient "0.4"))
;; Keywords: tools, matching

;;; Commentary:

;; An Emacs interface to nngrep and nnindex.
;;
;; M-x nngrep asks for a query and searches the index covering the current
;; directory, offering to build one if there is none.  The results buffer
;; shows each chunk highlighted like its file:
;;
;;   RET    visit the line at point      n, p   next or previous chunk
;;   C-o    show it in another window    TAB    fold or unfold the chunk
;;   s      search for something else    S-TAB  fold or unfold every chunk
;;   g      search again                 e      add a query
;;   r      toggle reranking             d      search another directory
;;   I      update the index             +, -   more or fewer results
;;   ?      menu of all options
;;
;; M-g n and M-g p step through the results from any buffer.  M-x nngrep-menu
;; opens the menu of options before searching.

;;; Code:

(require 'cl-lib)
(require 'seq)
(require 'transient)
(require 'project)
(require 'pulse)

(defgroup nngrep nil
  "Semantic search with nngrep."
  :group 'tools
  :prefix "nngrep-")

(defcustom nngrep-program "nngrep"
  "The nngrep executable."
  :type 'string)

(defcustom nngrep-index-program "nnindex"
  "The nnindex executable."
  :type 'string)

(defcustom nngrep-arguments nil
  "Options passed to nngrep by default, such as (\"--top=20\" \"--rerank\")."
  :type '(repeat string))

(defcustom nngrep-fontify t
  "Whether to highlight chunks with the major mode of their file."
  :type 'boolean)

(defface nngrep-score '((t :inherit font-lock-constant-face))
  "Face for the score of a chunk.")

(defface nngrep-path '((t :inherit font-lock-function-name-face))
  "Face for the path of a chunk's file.")

(defface nngrep-lines '((t :inherit shadow))
  "Face for the line range of a chunk.")

(defface nngrep-line-number '((t :inherit line-number))
  "Face for line numbers in front of chunk text.")

(defface nngrep-note '((t :inherit shadow :slant italic))
  "Face for notes such as a file having changed since indexing.")

(defvar nngrep-history nil
  "Queries given to `nngrep'.")

(cl-defstruct (nngrep-hit (:constructor nngrep-hit-create))
  "A chunk found by nngrep."
  score file start end stale lines)

(defvar-local nngrep--queries nil
  "Queries of the search shown in the buffer.")

(defvar-local nngrep--args nil
  "Options of the search shown in the buffer.")

(defvar-local nngrep--process nil
  "Running nngrep process, if any.")

(defvar-local nngrep--unvisited nil
  "Has no result been shown since the search finished?")

(defvar-local nngrep--status nil
  "Status of the search, shown in the header line.")

(defvar nngrep--index-process nil
  "Running nnindex process, if any.")

;;;; Searching

(defun nngrep--index-root (dir)
  "Return the root of the index covering DIR, or nil if there is none."
  (when-let* ((root (locate-dominating-file
                     dir (lambda (d) (file-exists-p
                                      (expand-file-name ".nngrep/index.json" d))))))
    (expand-file-name root)))

(defun nngrep--default-directory ()
  "Return the directory to search by default."
  (expand-file-name
   (or (nngrep--index-root default-directory)
       (when-let* ((project (project-current))) (project-root project))
       default-directory)))

(defun nngrep--read-query (&optional prompt initial)
  "Read a query with PROMPT, starting from INITIAL or the active region."
  (let ((initial (or initial
                     (when (use-region-p)
                       (buffer-substring-no-properties
                        (region-beginning) (region-end))))))
    (string-trim (read-string (or prompt "nngrep: ") initial 'nngrep-history))))

(defun nngrep--search (queries dir args)
  "Search DIR for the chunks best matching QUERIES, using nngrep options ARGS.
Offers to index DIR first if no index covers it."
  (when (member "" queries)
    (user-error "Empty query"))
  (setq dir (file-name-as-directory (expand-file-name dir)))
  (cond
   ((nngrep--index-root dir)
    (let ((buffer (get-buffer-create "*nngrep*")))
      (with-current-buffer buffer
        (unless (derived-mode-p 'nngrep-mode)
          (nngrep-mode))
        (setq default-directory dir
              nngrep--queries queries
              nngrep--args args)
        (nngrep--start))
      (setq next-error-last-buffer buffer)
      (pop-to-buffer buffer)))
   ((y-or-n-p (format "No index covers %s.  Index it now? " (abbreviate-file-name dir)))
    (nngrep--index dir (lambda () (nngrep--search queries dir args))))))

(defun nngrep--command ()
  "Return the nngrep command line for the search in the current buffer."
  `(,nngrep-program "--heading" "--line-number" "--full" "--color=never"
                    ,@nngrep--args
                    ,@(mapcar (lambda (q) (concat "--query=" q)) nngrep--queries)))

(defun nngrep--start ()
  "Run the search of the current buffer and show its results when done."
  (nngrep--stop)
  (let* ((buffer (current-buffer))
         (stdout (generate-new-buffer " *nngrep-stdout*"))
         (stderr (generate-new-buffer " *nngrep-stderr*"))
         (stderr-pipe (make-pipe-process :name "nngrep-stderr" :buffer stderr
                                         :noquery t :sentinel #'ignore))
         (started (float-time)))
    (setq nngrep--status "searching…")
    (let ((inhibit-read-only t))
      (erase-buffer)
      (insert (propertize "Searching…\n" 'font-lock-face 'nngrep-note)))
    (setq nngrep--process
          (make-process
           :name "nngrep"
           :buffer stdout
           :command (nngrep--command)
           :stderr stderr-pipe
           :noquery t
           :connection-type 'pipe
           :sentinel
           (lambda (process _event)
             (unless (process-live-p process)
               (accept-process-output stderr-pipe 0.1 nil t)
               (let ((output (with-current-buffer stdout (buffer-string)))
                     (errors (with-current-buffer stderr (string-trim (buffer-string))))
                     (status (process-exit-status process)))
                 (delete-process stderr-pipe)
                 (kill-buffer stdout)
                 (kill-buffer stderr)
                 (when (buffer-live-p buffer)
                   (with-current-buffer buffer
                     (when (eq process nngrep--process)
                       (setq nngrep--process nil)
                       (if (eq (process-status process) 'exit)
                           (nngrep--show (nngrep--parse output) errors status
                                         (- (float-time) started))
                         (setq nngrep--status "stopped")
                         (force-mode-line-update))))))))))))

(defun nngrep--stop ()
  "Stop the running search of the current buffer, if any."
  (when (process-live-p nngrep--process)
    (let ((process nngrep--process))
      (setq nngrep--process nil)
      (delete-process process))))

(defun nngrep--parse (output)
  "Return the hits in OUTPUT, printed by nngrep with --heading and -n."
  (let (hits hit)
    (dolist (line (split-string output "\n"))
      (cond
       ((string-match "\\`\\([0-9]+\\):\\(.*\\)\\'" line)
        (when hit
          (push (cons (string-to-number (match-string 1 line)) (match-string 2 line))
                (nngrep-hit-lines hit))))
       ((string-match "\\`\\(-?[0-9.]+\\)  \\(.+\\):\\([0-9]+\\)-\\([0-9]+\\)\\( (changed since indexing)\\)?\\'"
                      line)
        (setq hit (nngrep-hit-create
                   :score (string-to-number (match-string 1 line))
                   :file (match-string 2 line)
                   :start (string-to-number (match-string 3 line))
                   :end (string-to-number (match-string 4 line))
                   :stale (and (match-beginning 5) t)))
        (push hit hits))))
    (dolist (hit hits)
      (setf (nngrep-hit-lines hit) (nreverse (nngrep-hit-lines hit))))
    (nreverse hits)))

;;;; Results buffer

(defun nngrep--show (hits errors status seconds)
  "Fill the buffer with HITS and the ERRORS printed by nngrep.
STATUS is nngrep's exit status and SECONDS how long the search took."
  (let ((inhibit-read-only t)
        (width (length (number-to-string
                        (apply #'max 0 (mapcar #'nngrep-hit-end hits))))))
    (erase-buffer)
    (unless (string-empty-p errors)
      (insert (propertize errors 'font-lock-face 'error) "\n\n")
      (when (string-match-p "no index found" errors)
        (insert (substitute-command-keys
                 "Press \\[nngrep-index-and-search] to build the index.\n\n"))))
    (when (and (null hits) (/= status 2))
      (insert (propertize "No results.\n" 'font-lock-face 'nngrep-note)))
    (dolist (hit hits)
      (nngrep--insert-hit hit width))
    (goto-char (or (nngrep--first-hit) (point-min)))
    (setq nngrep--unvisited t))
  (setq nngrep--status
        (format "%d %s in %.1f s" (length hits)
                (if (= (length hits) 1) "result" "results") seconds))
  (force-mode-line-update))

(defun nngrep--insert-hit (hit width)
  "Insert HIT, with line numbers padded to WIDTH."
  (let ((start (point)))
    (insert (propertize (format "%.3f" (nngrep-hit-score hit)) 'font-lock-face 'nngrep-score)
            "  "
            (propertize (nngrep-hit-file hit) 'font-lock-face 'nngrep-path)
            (propertize (format ":%d-%d" (nngrep-hit-start hit) (nngrep-hit-end hit))
                        'font-lock-face 'nngrep-lines))
    (when (nngrep-hit-stale hit)
      (insert (propertize "  changed since indexing" 'font-lock-face 'nngrep-note)))
    (add-text-properties start (point) '(nngrep-heading t mouse-face highlight
                                         help-echo "mouse-2: visit this chunk"))
    (insert "\n")
    (let ((text (nngrep--fontify
                 (mapconcat #'cdr (nngrep-hit-lines hit) "\n")
                 (nngrep-hit-file hit))))
      (cl-loop for (number . _) in (nngrep-hit-lines hit)
               for line in (split-string text "\n")
               do (insert (propertize (concat (string-pad (number-to-string number) width nil t) "  ")
                                      'font-lock-face 'nngrep-line-number)
                          (propertize line 'nngrep-line number 'nngrep-column (+ width 2))
                          "\n")))
    (insert "\n")
    (put-text-property start (point) 'nngrep-hit hit)))

(defvar nngrep--fontify-buffers nil
  "Alist of major modes and the buffers used to highlight chunks with them.")

(defun nngrep--file-mode (file)
  "Return the major mode `auto-mode-alist' gives for FILE, or nil."
  (let ((mode (assoc-default file auto-mode-alist #'string-match-p)))
    (when (consp mode)
      (setq mode (car mode)))
    (when mode
      (setq mode (if (fboundp 'major-mode-remap)
                     (major-mode-remap mode)
                   (alist-get mode major-mode-remap-alist mode))))
    (and (functionp mode) mode)))

(defun nngrep--fontify (text file)
  "Return TEXT highlighted as in FILE's major mode, if `nngrep-fontify' is set."
  (let ((mode (and nngrep-fontify (nngrep--file-mode file))))
    (or (and mode
             (with-demoted-errors "nngrep: cannot highlight: %S"
               (let ((buffer (alist-get mode nngrep--fontify-buffers)))
                 (unless (buffer-live-p buffer)
                   (setq buffer (generate-new-buffer (format " *nngrep-%s*" mode)))
                   (with-current-buffer buffer
                     (delay-mode-hooks (funcall mode)))
                   (setf (alist-get mode nngrep--fontify-buffers) buffer))
                 (with-current-buffer buffer
                   (let ((inhibit-modification-hooks t))
                     (erase-buffer)
                     (insert text))
                   (font-lock-ensure)
                   (nngrep--face-to-font-lock-face (buffer-string))))))
        text)))

(defun nngrep--face-to-font-lock-face (string)
  "Return STRING with its `face' properties turned into `font-lock-face'."
  (let ((pos 0))
    (while (< pos (length string))
      (let ((next (next-single-property-change pos 'face string (length string)))
            (face (get-text-property pos 'face string)))
        (when face
          (put-text-property pos next 'font-lock-face face string))
        (setq pos next)))
    (remove-text-properties 0 (length string) '(face nil) string)
    string))

(defun nngrep--header-line ()
  "Return the header line of the results buffer."
  (concat
   " "
   (mapconcat (lambda (q) (propertize q 'face 'bold)) nngrep--queries
              (propertize " | " 'face 'shadow))
   (propertize (concat "  in " (abbreviate-file-name default-directory)) 'face 'shadow)
   (when nngrep--args
     (concat "  " (string-join nngrep--args " ")))
   (when nngrep--status
     (propertize (concat "  " nngrep--status) 'face 'shadow))))

(defvar-keymap nngrep-mode-map
  :parent special-mode-map
  "RET" #'nngrep-visit
  "C-o" #'nngrep-display
  "<mouse-2>" #'nngrep-mouse-visit
  "n" #'nngrep-next-result
  "p" #'nngrep-previous-result
  "TAB" #'nngrep-toggle-fold
  "<backtab>" #'nngrep-toggle-fold-all
  "g" #'nngrep-search-again
  "s" #'nngrep-new-query
  "e" #'nngrep-add-query
  "d" #'nngrep-change-directory
  "r" #'nngrep-toggle-rerank
  "+" #'nngrep-more-results
  "-" #'nngrep-fewer-results
  "I" #'nngrep-index-and-search
  "?" #'nngrep-menu
  "C-c C-k" #'nngrep-kill)

(define-derived-mode nngrep-mode special-mode "nngrep"
  "Major mode for nngrep search results.

\\{nngrep-mode-map}"
  (setq-local header-line-format '(:eval (nngrep--header-line)))
  (setq-local mode-line-process '(nngrep--process ":run"))
  (setq-local next-error-function #'nngrep--next-error)
  (setq-local revert-buffer-function (lambda (&rest _) (nngrep-search-again)))
  (setq-local truncate-lines t)
  (add-to-invisibility-spec '(nngrep . t))
  (add-hook 'kill-buffer-hook #'nngrep--stop nil t))

;;;; Commands in the results buffer

(defun nngrep--hit-at (&optional pos)
  "Return the hit at POS (default point), or signal an error."
  (or (get-text-property (or pos (point)) 'nngrep-hit)
      (user-error "No result here")))

(defun nngrep--hit-start (&optional pos)
  "Return the position of the heading of the hit at POS."
  (setq pos (or pos (point)))
  (if (and (> pos (point-min))
           (eq (get-text-property (1- pos) 'nngrep-hit)
               (get-text-property pos 'nngrep-hit)))
      (or (previous-single-property-change pos 'nngrep-hit) (point-min))
    pos))

(defun nngrep--location ()
  "Return the line and column in the file of the text at point."
  (if-let* ((line (get-text-property (point) 'nngrep-line)))
      (cons line (max 0 (- (current-column)
                           (get-text-property (point) 'nngrep-column))))
    (cons (nngrep-hit-start (nngrep--hit-at)) 0)))

(defun nngrep--line-position (line &optional end)
  "Return the start of LINE in the whole buffer, or its end if END is set."
  (save-restriction
    (widen)
    (save-excursion
      (goto-char (point-min))
      (forward-line (1- line))
      (if end (line-end-position) (point)))))

(defun nngrep--goto (display)
  "Show the location at point in its file with DISPLAY and return the window.
DISPLAY is a function like `pop-to-buffer', given the file's buffer."
  (setq nngrep--unvisited nil)
  (let* ((hit (nngrep--hit-at))
         (location (nngrep--location))
         (file (expand-file-name (nngrep-hit-file hit)))
         (buffer (or (find-buffer-visiting file)
                     (if (file-exists-p file)
                         (find-file-noselect file)
                       (user-error "%s no longer exists" (nngrep-hit-file hit)))))
         (window (funcall display buffer)))
    (with-selected-window window
      (let ((start (nngrep--line-position (nngrep-hit-start hit)))
            (end (nngrep--line-position (nngrep-hit-end hit) t)))
        (when (or (< start (point-min)) (> end (point-max)))
          (widen))
        (goto-char (nngrep--line-position (car location)))
        (move-to-column (cdr location))
        (unless (and (pos-visible-in-window-p start window)
                     (pos-visible-in-window-p end window))
          (set-window-start window (save-excursion
                                     (goto-char start)
                                     (forward-line -2)
                                     (point))))
        (pulse-momentary-highlight-region start end)))
    window))

(defun nngrep--display-other (buffer)
  "Display BUFFER in another window without selecting it."
  (display-buffer buffer '(nil (inhibit-same-window . t))))

(defun nngrep-visit ()
  "Visit the file line at point in another window."
  (interactive)
  (select-window (nngrep--goto #'nngrep--display-other)))

(defun nngrep-display ()
  "Show the file line at point in another window."
  (interactive)
  (nngrep--goto #'nngrep--display-other))

(defun nngrep-mouse-visit (event)
  "Visit the file line clicked on with EVENT in another window."
  (interactive "e")
  (mouse-set-point event)
  (nngrep-visit))

(defun nngrep--first-hit ()
  "Return the position of the first hit, or nil if there is none."
  (if (get-text-property (point-min) 'nngrep-hit)
      (point-min)
    (next-single-property-change (point-min) 'nngrep-hit)))

(defun nngrep-next-result (&optional n no-display)
  "Move to the Nth next result and show it in another window.
With NO-DISPLAY, only move."
  (interactive "p")
  (setq n (or n 1))
  (let ((pos (point)))
    (if (< n 0)
        (dotimes (_ (- n))
          (setq pos (nngrep--hit-start pos))
          (unless (and (> pos (point-min)) (get-text-property (1- pos) 'nngrep-hit))
            (user-error "No previous result"))
          (setq pos (nngrep--hit-start (1- pos))))
      (dotimes (_ n)
        (setq pos (next-single-property-change pos 'nngrep-hit))
        (unless (and pos (get-text-property pos 'nngrep-hit))
          (user-error "No more results"))))
    (goto-char pos))
  (unless no-display
    (nngrep-display)))

(defun nngrep-previous-result (&optional n)
  "Move to the Nth previous result and show it in another window."
  (interactive "p")
  (nngrep-next-result (- (or n 1))))

(defun nngrep--next-error (n &optional reset)
  "Visit the Nth next result, starting from the first if RESET is non-nil.
Used as `next-error-function'."
  (if (not (or reset nngrep--unvisited))
      (nngrep-next-result n t)
    (goto-char (or (nngrep--first-hit) (user-error "No results")))
    (when (> n 1)
      (nngrep-next-result (1- n) t)))
  (when-let* ((window (get-buffer-window (current-buffer))))
    (set-window-point window (point)))
  (let ((results (current-buffer)))
    (select-window
     (nngrep--goto (lambda (buffer)
                     (if (eq (window-buffer) results)
                         (nngrep--display-other buffer)
                       (display-buffer buffer '(display-buffer-same-window))))))))

(defun nngrep-toggle-fold (&optional pos)
  "Hide or show the text of the chunk at POS (default point)."
  (interactive)
  (nngrep--hit-at pos)
  (let* ((start (nngrep--hit-start pos))
         (body (save-excursion (goto-char start) (line-beginning-position 2)))
         (end (or (next-single-property-change start 'nngrep-hit) (point-max)))
         (fold (seq-find (lambda (o) (overlay-get o 'nngrep-fold))
                         (overlays-at body))))
    (if fold
        (delete-overlay fold)
      (let ((overlay (make-overlay body (max body (1- end)))))
        (overlay-put overlay 'nngrep-fold t)
        (overlay-put overlay 'invisible 'nngrep)
        (overlay-put overlay 'isearch-open-invisible #'delete-overlay)))
    (goto-char start)))

(defun nngrep-toggle-fold-all ()
  "Hide the text of every chunk, or show it all if any is hidden."
  (interactive)
  (let ((folds (seq-filter (lambda (o) (overlay-get o 'nngrep-fold))
                           (overlays-in (point-min) (point-max)))))
    (if folds
        (mapc #'delete-overlay folds)
      (save-excursion
        (let ((pos (nngrep--first-hit)))
          (while (and pos (get-text-property pos 'nngrep-hit))
            (nngrep-toggle-fold pos)
            (setq pos (next-single-property-change pos 'nngrep-hit))))))))

(defun nngrep--results-buffer ()
  "Signal an error unless the current buffer shows nngrep results."
  (unless (derived-mode-p 'nngrep-mode)
    (user-error "Not in an nngrep results buffer")))

(defun nngrep-search-again ()
  "Run the search again."
  (interactive)
  (nngrep--results-buffer)
  (nngrep--search nngrep--queries default-directory nngrep--args))

(defun nngrep-new-query (query)
  "Search for QUERY instead, in the same directory and with the same options."
  (interactive (list (nngrep--read-query nil (car nngrep--queries))))
  (nngrep--results-buffer)
  (nngrep--search (list query) default-directory nngrep--args))

(defun nngrep-add-query (query)
  "Search for QUERY too.  Each chunk is scored by its best matching query."
  (interactive (list (nngrep--read-query "Add query: ")))
  (nngrep--results-buffer)
  (nngrep--search (append nngrep--queries (list query)) default-directory nngrep--args))

(defun nngrep-change-directory (dir)
  "Run the search again in DIR."
  (interactive (list (read-directory-name "Search in: ")))
  (nngrep--results-buffer)
  (nngrep--search nngrep--queries dir nngrep--args))

(defun nngrep--top (args)
  "Return the number of results ARGS ask for."
  (string-to-number (or (transient-arg-value "--top=" args) "10")))

(defun nngrep--with-top (args top)
  "Return ARGS asking for TOP results."
  (append (seq-remove (lambda (a) (string-prefix-p "--top=" a)) args)
          (list (format "--top=%d" top))))

(defun nngrep-more-results (&optional n)
  "Run the search again with N more results (10 without a prefix argument)."
  (interactive "P")
  (nngrep--results-buffer)
  (let ((step (if n (prefix-numeric-value n) 10)))
    (nngrep--search nngrep--queries default-directory
                    (nngrep--with-top nngrep--args (+ (nngrep--top nngrep--args) step)))))

(defun nngrep-fewer-results (&optional n)
  "Run the search again with N fewer results (10 without a prefix argument)."
  (interactive "P")
  (nngrep--results-buffer)
  (let ((step (if n (prefix-numeric-value n) 10)))
    (nngrep--search nngrep--queries default-directory
                    (nngrep--with-top nngrep--args
                                      (max 1 (- (nngrep--top nngrep--args) step))))))

(defun nngrep-toggle-rerank ()
  "Run the search again with reranking turned on or off."
  (interactive)
  (nngrep--results-buffer)
  (nngrep--search nngrep--queries default-directory
                  (if (member "--rerank" nngrep--args)
                      (seq-remove (lambda (a) (or (equal a "--rerank")
                                                  (string-prefix-p "--rerank-depth=" a)))
                                  nngrep--args)
                    (append nngrep--args '("--rerank")))))

(defun nngrep-kill ()
  "Stop the running search."
  (interactive)
  (nngrep--results-buffer)
  (nngrep--stop)
  (setq nngrep--status "stopped")
  (force-mode-line-update))

;;;; Indexing

(defun nngrep--index (dir &optional then)
  "Build or update the index of DIR with nnindex, then call THEN if it succeeds."
  (when (process-live-p nngrep--index-process)
    (user-error "nnindex is already running"))
  (setq dir (file-name-as-directory (expand-file-name dir)))
  (let ((buffer (get-buffer-create "*nnindex*"))
        (started (float-time)))
    (with-current-buffer buffer
      (unless (derived-mode-p 'special-mode)
        (special-mode))
      (setq default-directory dir)
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (format "Indexing %s\n\n" (abbreviate-file-name dir))))
      (setq-local mode-line-process '(nngrep--index-process ":run")))
    (setq nngrep--index-process
          (make-process
           :name "nnindex"
           :buffer buffer
           :command (list nngrep-index-program dir)
           :connection-type 'pipe
           :filter (lambda (process string)
                     (when (buffer-live-p (process-buffer process))
                       (with-current-buffer (process-buffer process)
                         (let ((inhibit-read-only t))
                           (save-excursion
                             (goto-char (point-max))
                             (insert string))))))
           :sentinel
           (lambda (process _event)
             (unless (process-live-p process)
               (let ((ok (and (eq (process-status process) 'exit)
                              (zerop (process-exit-status process)))))
                 (setq nngrep--index-process nil)
                 (when (buffer-live-p buffer)
                   (with-current-buffer buffer
                     (let ((inhibit-read-only t))
                       (goto-char (point-max))
                       (insert (format "\n%s after %.0f s\n"
                                       (if ok "Done" "Failed")
                                       (- (float-time) started))))
                     (force-mode-line-update)))
                 (if (not ok)
                     (message "nnindex failed; see the *nnindex* buffer")
                   (message "Indexed %s" (abbreviate-file-name dir))
                   (when then (funcall then))))))))
    (display-buffer buffer)
    (message "Indexing %s…" (abbreviate-file-name dir))))

;;;###autoload
(defun nngrep-index (dir)
  "Build or update the nngrep index of DIR in the background."
  (interactive (list (read-directory-name
                      "Index directory: "
                      (or (nngrep--index-root default-directory)
                          (when-let* ((project (project-current))) (project-root project))))))
  (nngrep--index dir))

(defun nngrep-index-and-search ()
  "Update the index covering the search, then run the search again."
  (interactive)
  (nngrep--results-buffer)
  (let ((buffer (current-buffer)))
    (nngrep--index (or (nngrep--index-root default-directory) default-directory)
                   (lambda ()
                     (when (buffer-live-p buffer)
                       (with-current-buffer buffer
                         (nngrep-search-again)))))))

;;;; Entry points and menu

(defun nngrep--menu-args ()
  "Return the options set in `nngrep-menu', or the current ones outside it."
  (if (eq transient-current-command 'nngrep-menu)
      (transient-args 'nngrep-menu)
    (nngrep--current-args)))

(defun nngrep--current-args ()
  "Return the options of the search in the current buffer, or the defaults."
  (if (derived-mode-p 'nngrep-mode) nngrep--args nngrep-arguments))

;;;###autoload
(defun nngrep (query &optional dir)
  "Search for QUERY in DIR with nngrep.
DIR defaults to the root of the index covering the current directory, or
else the current project.  With a prefix argument, ask for DIR."
  (interactive
   (list (nngrep--read-query)
         (when current-prefix-arg
           (read-directory-name "Search in: " (nngrep--default-directory)))))
  (nngrep--search (list query) (or dir (nngrep--default-directory)) (nngrep--menu-args)))

(defun nngrep-menu-search-in (query dir)
  "Search for QUERY in DIR with the options set in the menu."
  (interactive
   (list (nngrep--read-query)
         (read-directory-name "Search in: " (nngrep--default-directory))))
  (nngrep--search (list query) dir (nngrep--menu-args)))

(defun nngrep-menu-search-again ()
  "Run the search shown in the current buffer again with the menu's options."
  (interactive)
  (nngrep--results-buffer)
  (nngrep--search nngrep--queries default-directory (nngrep--menu-args)))

(defun nngrep-menu-add-query (query)
  "Search for QUERY too, with the menu's options."
  (interactive (list (nngrep--read-query "Add query: ")))
  (nngrep--results-buffer)
  (nngrep--search (append nngrep--queries (list query)) default-directory (nngrep--menu-args)))

(defun nngrep--in-results-p ()
  "Is the current buffer an nngrep results buffer?"
  (derived-mode-p 'nngrep-mode))

;;;###autoload (autoload 'nngrep-menu "nngrep" nil t)
(transient-define-prefix nngrep-menu ()
  "Set nngrep's options and search."
  :init-value (lambda (obj) (oset obj value (nngrep--current-args)))
  ["Options"
   ("-k" "Number of results" "--top=" :class transient-option
    :reader transient-read-number-N+)
   ("-m" "Most chunks per file" "--max-count=" :class transient-option
    :reader transient-read-number-N+)
   ("-s" "Minimum score" "--min-score=" :class transient-option)
   ("-r" "Rerank" "--rerank")
   ("-d" "Chunks to rerank" "--rerank-depth=" :class transient-option
    :reader transient-read-number-N+)
   ("-g" "Glob" "--glob=" :class transient-option :multi-value repeat)
   ("-t" "Only file type" "--type=" :class transient-option :multi-value repeat)
   ("-T" "Skip file type" "--type-not=" :class transient-option :multi-value repeat)
   ("-c" "Run on the CPU" "--cpu")]
  [["Search"
    ("s" "New search" nngrep)
    ("S" "New search in…" nngrep-menu-search-in)
    ("g" "Search again" nngrep-menu-search-again :if nngrep--in-results-p)
    ("e" "Add a query" nngrep-menu-add-query :if nngrep--in-results-p)]
   ["Index"
    ("i" "Index a directory" nngrep-index)]])

(provide 'nngrep)
;;; nngrep.el ends here

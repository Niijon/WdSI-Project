# Description of project

## Program
We can display how program works by showcasing simple steps inside of it which can be translated into functions in main:
1. Loading annotations - **GetAnnotationData**
2. Displaying objects quantity - **CheckQuantity**
3. Printing loaded training annotations - **PrintAnnotations**
4. Loading images data - **LoadData**
5. Balancing data - **BalanceData**
6. Either loading voc.npy data file or learning with bag of views usage - **LearnBoVW**
7. Extracting descriptors and adding them to training data - **ExtractFeatures**
8. Training on our training dataset and then returning rainforest - **Train**
9. Printing test annotations - **PrintAnnotations**
10. Loading testing dataset - **LoadData**
11. Balancing testing dataset - **BalanceData**
12. Extracting descriptors and adding them to testing dataset - **ExtractFeatures**
13. Predict objects using rainforest created with dataset on testing dataset - **Predict**
14. Evaluating results of prediction - **Predict**

That's how we can shortly describe how code works. What is worth mentioning is that first approach was made with only reading images. What followed was that results ware clearly having low accuraccy(75%). After seeing that during loading images process every object inside of dataset was extracted from image so that accuraccy would increase. That indeed happened and source of this issue was probably very little amount of actual crosswalk signs inside of data. After adding object extraction to training dataset accuracy increased to almost 90%.

*****
## Analiza wprowadzanych zmian
Kroki które podjęto:
- Pierwszy program tworzony dla wykrywania 4 możliwych przypadków znaków został dostosowany do programu operującego na binarnym rozróżnianiu znaków tak aby spełniał on warunki projektu.
- Po stworzeniu pierwszej wersji programu starano się jak najlepiej dobrać bazę tak aby prezycja była relatywnie wysoka.
- Drugi program rozróżniał binarnie jednak z małą dokładnością co wiązało się z małą ilością znaków w bazie i tym że posiadaliśmy wiele znaków na jednym obrazie przez co gorzej działała identyfikacja znaków. W związku z tym wprowadzono opcje wycinania poszczególnych obiektów ze zdjęcia i uczenia się na ich podstawie.
- Trzeci program zaczął odrzucać przypadki pasów których powierzchnia zajmowała mniej niż 0.1 całego obrazu. Jednak konsekwencją tego było to że zabieg ten doprowadził do znaczącego zmniejszenia się obiektów w bazie uczącej i testowej. 
- Czwarty program posiadał powyżej wymieniony zabieg tylko dla bazy testowej jednak to z kolei doprowadzało do 100% dokładności jeżeli chodzi o wnioskowanie o znakach. Sama baza znaków testowych zmalało bowiem 10-cio krotnie i w ostateczności posiadaliśmy tylko znaki które dawały wynik prawdziwy.
- Ostatecznie zakomentowano fragmenty kodu tak abyśmy posiadali bazę bez sprawdzania warunku powierzchni, powrócono zatem do wersji drugiej.
- Po poprawie warunku tak aby odpowiadał on własności powierzchni pola naszego obrazka i odkomentowaniu fragmentu kodu który odrzucał zdjęcia które posiadały mniejszą powierzchnie niż 0.01 naszego obrazka uzyskaliśmy skuteczność w klasyfikacji równą 94.5%.
- Przetstowano również co się stanie gdy zastosujemy warunek prosty z instrukcji który zakłada że obiekt znajdujący się na obrazku musi posiadać szerokość oraz wysokość większą niż 0.1 szerokości i wysokości obrazka. W tym przypadku otrzymujemy większą dokładność naszego algorytmu która wzrasta do około 95%

Po poprawie warunku ilość obiektów klasyfikowanych wzrosła z 4 do 200. Łączna ilość obiektów na obrazkach dla zbioru testowego to 399. Oznacza to że ponad połowa naszych elementów nie jest w ogóle uwzględniania. Jeżeli uwzględnimy całość to nasza dokładność spada z około 94.5% do około 90%.

Dla przypadku w którym uwzględniamy warunek wysokość a nie pola obrazka otrzymujemy większą dokładność która wzrasta do 95% co daje nam dokładność większą o około 0.5 punkta procentowego. Można również zauważyć to że ilość obiektów spełniająca ten warunek wynosi 161 co daje nam łącznie 39 mniej elementów w zbiorze testowym niż w przypadku pola, można zatem wydedukować że otrzymujemy spadek w ilości zbioru danych równy około 10% całego zbioru. Warto jest zatem rozważyć czy bardziej opłaca się zachować większą ilość przypadków uwzględnionych przy lekko mniejszej dokładności czy uwzględnić te przypadki kosztem mniejszej dokładności, ponieważ dokładność jak można zauważyć jest ściśle związana z ilością danych w zbiorze gdy posiadamy więcej obrazów o mniejszej wielkości.


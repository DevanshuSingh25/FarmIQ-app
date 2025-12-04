import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUpDown } from "lucide-react";
import { MarketPrice } from "@/services/marketPricesService";
import { PaginationState, SortState } from "@/pages/MarketPrices";

interface PricesTableProps {
  prices: MarketPrice[];
  loading: boolean;
  pagination: PaginationState;
  sort: SortState;
  onPageChange: (page: number) => void;
  onSortChange: (field: keyof MarketPrice) => void;
}

export const PricesTable = ({
  prices,
  loading,
  pagination,
  sort,
  onPageChange,
  onSortChange
}: PricesTableProps) => {

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    });
  };

  const getSortIcon = (field: keyof MarketPrice) => {
    if (sort.field !== field) {
      return <ArrowUpDown className="h-4 w-4 opacity-50" />;
    }
    return (
      <ArrowUpDown className={`h-4 w-4 ${sort.direction === 'asc' ? 'rotate-180' : ''}`} />
    );
  };

  const renderTableHeader = () => (
    <TableHeader>
      <TableRow>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => onSortChange('state')}
        >
          <div className="flex items-center gap-2">
            State
            {getSortIcon('state')}
          </div>
        </TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => onSortChange('district')}
        >
          <div className="flex items-center gap-2">
            District/Mandi
            {getSortIcon('district')}
          </div>
        </TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => onSortChange('market')}
        >
          <div className="flex items-center gap-2">
            Market
            {getSortIcon('market')}
          </div>
        </TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors"
          onClick={() => onSortChange('commodity')}
        >
          <div className="flex items-center gap-2">
            Commodity
            {getSortIcon('commodity')}
          </div>
        </TableHead>
        <TableHead>Variety</TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors text-right"
          onClick={() => onSortChange('min_price')}
        >
          <div className="flex items-center justify-end gap-2">
            Min Price
            {getSortIcon('min_price')}
          </div>
        </TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors text-right"
          onClick={() => onSortChange('max_price')}
        >
          <div className="flex items-center justify-end gap-2">
            Max Price
            {getSortIcon('max_price')}
          </div>
        </TableHead>
        <TableHead
          className="cursor-pointer hover:bg-muted/50 transition-colors text-right"
          onClick={() => onSortChange('modal_price')}
        >
          <div className="flex items-center justify-end gap-2">
            Modal Price
            {getSortIcon('modal_price')}
          </div>
        </TableHead>
      </TableRow>
    </TableHeader>
  );

  const renderSkeletonRows = () => (
    <TableBody>
      {Array.from({ length: 10 }).map((_, index) => (
        <TableRow key={index}>
          <TableCell><Skeleton className="h-4 w-24" /></TableCell>
          <TableCell><Skeleton className="h-4 w-28" /></TableCell>
          <TableCell><Skeleton className="h-4 w-28" /></TableCell>
          <TableCell><Skeleton className="h-4 w-20" /></TableCell>
          <TableCell><Skeleton className="h-4 w-16" /></TableCell>
          <TableCell><Skeleton className="h-4 w-20" /></TableCell>
          <TableCell><Skeleton className="h-4 w-20" /></TableCell>
          <TableCell><Skeleton className="h-4 w-20" /></TableCell>
        </TableRow>
      ))}
    </TableBody>
  );

  const renderEmptyState = () => (
    <TableBody>
      <TableRow>
        <TableCell colSpan={8} className="text-center py-12">
          <div className="text-muted-foreground">
            <div className="text-lg mb-2">📊 No prices found</div>
            <p>Try adjusting your filters or check back later for updated data.</p>
          </div>
        </TableCell>
      </TableRow>
    </TableBody>
  );

  const renderDataRows = () => (
    <TableBody>
      {prices.map((price, index) => (
        <TableRow key={`${price.state}-${price.district}-${price.commodity}-${index}`} className="hover:bg-muted/50 transition-colors">
          <TableCell>{price.state || '—'}</TableCell>
          <TableCell>{price.district || '—'}</TableCell>
          <TableCell>{price.market || price.district || '—'}</TableCell>
          <TableCell>
            <Badge variant="outline" className="bg-success/10 text-success border-success/20">
              {price.commodity || '—'}
            </Badge>
          </TableCell>
          <TableCell className="text-muted-foreground">{price.variety || '—'}</TableCell>
          <TableCell className="text-right font-mono">
            {price.min_price !== null ? formatPrice(price.min_price) : '—'}
          </TableCell>
          <TableCell className="text-right font-mono">
            {price.max_price !== null ? formatPrice(price.max_price) : '—'}
          </TableCell>
          <TableCell className="text-right font-mono font-semibold">
            {price.modal_price !== null ? formatPrice(price.modal_price) : '—'}
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  );

  const renderPagination = () => {
    const totalPages = Math.ceil(pagination.total / pagination.pageSize);
    const currentPage = pagination.page;

    return (
      <div className="flex items-center justify-between mt-6">
        <div className="text-sm text-muted-foreground">
          Page {currentPage} of {totalPages}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(1)}
            disabled={currentPage === 1}
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let page;
              if (totalPages <= 5) {
                page = i + 1;
              } else if (currentPage <= 3) {
                page = i + 1;
              } else if (currentPage >= totalPages - 2) {
                page = totalPages - 4 + i;
              } else {
                page = currentPage - 2 + i;
              }

              return (
                <Button
                  key={page}
                  variant={page === currentPage ? "default" : "outline"}
                  size="sm"
                  className="w-8 h-8 p-0"
                  onClick={() => onPageChange(page)}
                >
                  {page}
                </Button>
              );
            })}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(totalPages)}
            disabled={currentPage === totalPages}
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="border rounded-lg bg-card">
        <div className="overflow-x-auto">
          <Table>
            {renderTableHeader()}
            {loading ? renderSkeletonRows() : prices.length === 0 ? renderEmptyState() : renderDataRows()}
          </Table>
        </div>
      </div>

      {!loading && prices.length > 0 && renderPagination()}
    </div>
  );
};